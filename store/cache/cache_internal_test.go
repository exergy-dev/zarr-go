package cache

import (
	"context"
	"errors"
	"iter"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rkm/zarr-go/store"
	"github.com/rkm/zarr-go/store/mem"
)

// gatedStore wraps a store.Store, letting tests pause Get at a controlled
// point. Each call to Get first (optionally) blocks sending on entered —
// letting the test observe exactly when this call started, and how many
// have started — then performs the real underlying read (capturing
// whatever value is current AT THAT MOMENT), then (optionally) blocks on
// gate before returning that already-captured value. This ordering matters:
// gating before the read would let a concurrent Set that lands while the
// gate is closed be observed by the "stale" read once it finally executes,
// defeating tests that need a read known to have captured a value that a
// later Set/Delete makes stale.
type gatedStore struct {
	inner store.Store

	mu    sync.Mutex
	calls int

	gate    chan struct{} // if non-nil, Get blocks here (or on ctx.Done()) after the real read, before returning
	entered chan struct{} // if non-nil, Get blocks sending here before the real read

	// gates, if non-nil, overrides gate per call: the Nth Get (1-based)
	// blocks on gates[N-1] instead (nil = don't block), so a test can
	// release one specific in-flight fetch while others stay blocked. Must
	// be set before the first Get and cover every call the test makes.
	gates []chan struct{}
}

func newGatedStore(inner store.Store) *gatedStore {
	return &gatedStore{inner: inner}
}

func (g *gatedStore) Get(ctx context.Context, key string) ([]byte, error) {
	g.mu.Lock()
	g.calls++
	gate := g.gate
	if g.gates != nil {
		gate = g.gates[g.calls-1]
	}
	g.mu.Unlock()
	// Perform the real read BEFORE signaling entered, so a test that
	// receives from entered and then mutates the key can rely on this call
	// having already captured its (possibly soon-to-be-stale) value — a
	// send on entered before the read would only prove the call started,
	// racing the read itself against the test's mutation.
	val, err := g.inner.Get(ctx, key)
	if g.entered != nil {
		select {
		case g.entered <- struct{}{}:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return val, err
}

func (g *gatedStore) Calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func (g *gatedStore) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	return g.inner.GetRange(ctx, key, off, n)
}
func (g *gatedStore) Set(ctx context.Context, key string, value []byte) error {
	return g.inner.Set(ctx, key, value)
}
func (g *gatedStore) Delete(ctx context.Context, key string) error {
	return g.inner.Delete(ctx, key)
}
func (g *gatedStore) Exists(ctx context.Context, key string) (bool, error) {
	return g.inner.Exists(ctx, key)
}
func (g *gatedStore) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return g.inner.List(ctx, prefix)
}

var _ store.Store = (*gatedStore)(nil)

// TestCacheConcurrentMissCoalescesDeterministic proves N concurrent misses
// on the same key result in exactly one underlying Get, using
// testHookWaiterJoined (not a scheduling race) to know precisely when all
// N-1 non-leader goroutines have joined the flight before releasing it.
func TestCacheConcurrentMissCoalescesDeterministic(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("v")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate := make(chan struct{})
	gated.gate = gate
	c := New(Config{Underlying: gated})

	const n = 32
	var joined int32
	allJoined := make(chan struct{})
	testHookWaiterJoined = func() {
		if atomic.AddInt32(&joined, 1) == n-1 {
			close(allJoined)
		}
	}
	t.Cleanup(func() { testHookWaiterJoined = nil })

	var wg sync.WaitGroup
	results := make([][]byte, n)
	errs := make([]error, n)
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = c.Get(ctx, "k")
		}(i)
	}

	select {
	case <-allJoined:
	case <-time.After(5 * time.Second):
		t.Fatal("not all waiters joined the flight in time")
	}
	close(gate)
	wg.Wait()

	for i, e := range errs {
		if e != nil {
			t.Fatalf("goroutine %d: %v", i, e)
		}
		if string(results[i]) != "v" {
			t.Fatalf("goroutine %d: got %q want %q", i, results[i], "v")
		}
	}
	if got := gated.Calls(); got != 1 {
		t.Fatalf("underlying Get calls: got %d want 1", got)
	}
}

// TestCacheStaleReAdmitOnSet is the regression test for the core coherency
// bug: a Get that started before a concurrent Set must never let its
// (now-stale) result get admitted into the cache after the Set lands.
func TestCacheStaleReAdmitOnSet(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("old")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate := make(chan struct{})
	entered := make(chan struct{})
	gated.gate = gate
	gated.entered = entered
	c := New(Config{Underlying: gated})

	getDone := make(chan struct{})
	var getVal []byte
	var getErr error
	go func() {
		getVal, getErr = c.Get(ctx, "k")
		close(getDone)
	}()
	<-entered // leader's underlying.Get has started, about to observe "old"

	if err := c.Set(ctx, "k", []byte("new")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	close(gate) // let the leader's fetch (of "old") complete
	<-getDone

	if getErr != nil {
		t.Fatalf("Get: %v", getErr)
	}
	if string(getVal) != "old" {
		t.Fatalf("Get returned %q, want %q (a legitimate read/write race snapshot)", getVal, "old")
	}

	// Disarm gating: this verification Get is unrelated to the race being
	// tested and would otherwise deadlock trying to signal entered/gate
	// with nothing left listening.
	gated.entered = nil
	gated.gate = nil

	got, err := c.Get(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "new" {
		t.Fatalf("cache serves stale value: got %q, want %q", got, "new")
	}
	if got := gated.Calls(); got != 2 {
		t.Fatalf("underlying Get calls: got %d, want 2 (first miss + re-fetch after poison)", got)
	}
}

// TestCacheStaleReAdmitOnDelete mirrors TestCacheStaleReAdmitOnSet for
// Delete: a Get racing a Delete must not resurrect the deleted key in cache.
func TestCacheStaleReAdmitOnDelete(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("old")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate := make(chan struct{})
	entered := make(chan struct{})
	gated.gate = gate
	gated.entered = entered
	c := New(Config{Underlying: gated})

	getDone := make(chan struct{})
	var getVal []byte
	var getErr error
	go func() {
		getVal, getErr = c.Get(ctx, "k")
		close(getDone)
	}()
	<-entered

	if err := c.Delete(ctx, "k"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	close(gate)
	<-getDone

	if getErr != nil || string(getVal) != "old" {
		t.Fatalf("Get: got %q err=%v, want %q,nil (legitimate race snapshot)", getVal, getErr, "old")
	}

	// Disarm gating: this verification Get is unrelated to the race being
	// tested and would otherwise deadlock trying to signal entered/gate
	// with nothing left listening.
	gated.entered = nil
	gated.gate = nil

	if _, err := c.Get(ctx, "k"); !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("cache resurrected deleted key: Get = %v, want ErrKeyNotFound", err)
	}
}

// TestCacheLeaderCancelWaiterSucceeds checks that when the leader's own ctx
// is cancelled/times out mid-fetch, a still-live waiter retries (becoming
// the next leader with its own ctx) instead of also failing.
func TestCacheLeaderCancelWaiterSucceeds(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("v")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate := make(chan struct{})
	entered := make(chan struct{})
	gated.gate = gate
	gated.entered = entered
	c := New(Config{Underlying: gated})

	leaderCtx, cancel := context.WithCancel(ctx)
	leaderDone := make(chan struct{})
	var leaderErr error
	go func() {
		_, leaderErr = c.Get(leaderCtx, "k")
		close(leaderDone)
	}()
	<-entered // leader's underlying.Get call (#1) has started

	var joined int32
	waiterJoined := make(chan struct{})
	testHookWaiterJoined = func() {
		if atomic.AddInt32(&joined, 1) == 1 {
			close(waiterJoined)
		}
	}
	t.Cleanup(func() { testHookWaiterJoined = nil })

	waiterDone := make(chan struct{})
	var waiterVal []byte
	var waiterErr error
	go func() {
		waiterVal, waiterErr = c.Get(ctx, "k") // no deadline
		close(waiterDone)
	}()
	<-waiterJoined // waiter has joined the leader's flight

	cancel() // leader's ctx cancelled; call #1 returns ctx.Err() without consuming the gate
	select {
	case <-leaderDone:
	case <-time.After(5 * time.Second):
		t.Fatal("leader did not return after its ctx was cancelled")
	}
	if !errors.Is(leaderErr, context.Canceled) {
		t.Fatalf("leader err = %v, want context.Canceled", leaderErr)
	}

	select {
	case <-entered: // the waiter retried and became the new leader (call #2)
	case <-time.After(5 * time.Second):
		t.Fatal("waiter did not retry as the new leader")
	}
	close(gate) // let call #2 proceed
	select {
	case <-waiterDone:
	case <-time.After(5 * time.Second):
		t.Fatal("waiter did not complete after retry")
	}

	if waiterErr != nil {
		t.Fatalf("waiter err = %v, want nil", waiterErr)
	}
	if string(waiterVal) != "v" {
		t.Fatalf("waiter val = %q, want %q", waiterVal, "v")
	}
	if got := gated.Calls(); got != 2 {
		t.Fatalf("underlying Get calls = %d, want 2", got)
	}
}

// TestCacheWaiterOwnCtxCancelReturnsImmediately checks that a waiter whose
// own ctx is cancelled returns right away, without waiting for the leader.
func TestCacheWaiterOwnCtxCancelReturnsImmediately(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("v")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate := make(chan struct{})
	entered := make(chan struct{})
	gated.gate = gate
	gated.entered = entered
	c := New(Config{Underlying: gated})

	leaderDone := make(chan struct{})
	go func() {
		_, _ = c.Get(ctx, "k") // leader; released and drained at the end
		close(leaderDone)
	}()
	<-entered // leader blocked on gate

	var joined int32
	waiterJoined := make(chan struct{})
	testHookWaiterJoined = func() {
		if atomic.AddInt32(&joined, 1) == 1 {
			close(waiterJoined)
		}
	}
	t.Cleanup(func() { testHookWaiterJoined = nil })

	waiterCtx, cancel := context.WithCancel(ctx)
	waiterDone := make(chan struct{})
	var waiterErr error
	go func() {
		_, waiterErr = c.Get(waiterCtx, "k")
		close(waiterDone)
	}()
	<-waiterJoined

	cancel()
	select {
	case <-waiterDone:
	case <-time.After(5 * time.Second):
		t.Fatal("waiter did not return promptly after its own ctx was cancelled")
	}
	if !errors.Is(waiterErr, context.Canceled) {
		t.Fatalf("waiter err = %v, want context.Canceled", waiterErr)
	}

	close(gate) // let the leader finish so its goroutine doesn't leak past the test
	<-leaderDone
}

// TestCacheGetAfterSetDoesNotJoinStaleFlight is the regression test for a
// read-your-writes violation: a Get that STARTS after Set has returned must
// not join a fetch that began before the Set (and so captured the old
// value). It must start a fresh fetch and observe the write.
func TestCacheGetAfterSetDoesNotJoinStaleFlight(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("v1")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate := make(chan struct{})
	entered := make(chan struct{})
	gated.gate = gate
	gated.entered = entered
	c := New(Config{Underlying: gated})

	waiterJoined := make(chan struct{}, 1)
	testHookWaiterJoined = func() { waiterJoined <- struct{}{} }
	t.Cleanup(func() { testHookWaiterJoined = nil })

	g1Done := make(chan struct{})
	var g1Val []byte
	var g1Err error
	go func() {
		g1Val, g1Err = c.Get(ctx, "k")
		close(g1Done)
	}()
	<-entered // leader's underlying Get has captured "v1"

	if err := c.Set(ctx, "k", []byte("v2")); err != nil {
		t.Fatalf("Set: %v", err)
	}

	g2Done := make(chan struct{})
	var g2Val []byte
	var g2Err error
	go func() {
		g2Val, g2Err = c.Get(ctx, "k")
		close(g2Done)
	}()
	// The second Get either joins the stale flight (the bug) or starts its
	// own fetch, which reads "v2" and signals entered. Exactly one happens.
	select {
	case <-waiterJoined:
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("second Get neither joined a flight nor started a fetch")
	}
	close(gate)
	<-g1Done
	<-g2Done

	// The first Get started before the Set completed, so the old value is a
	// legitimate (linearizable) result for it.
	if g1Err != nil || string(g1Val) != "v1" {
		t.Fatalf("first Get: got %q err=%v, want %q,nil", g1Val, g1Err, "v1")
	}
	if g2Err != nil {
		t.Fatalf("second Get: %v", g2Err)
	}
	if string(g2Val) != "v2" {
		t.Fatalf("Get started after Set returned %q, want %q (read-your-writes)", g2Val, "v2")
	}
}

// TestCacheStaleLeaderKeepsFreshFlight checks that when a flight detached
// by Set finishes while a fresh flight for the same key is still running,
// the stale leader does not remove the fresh flight from the tracker: a
// later Get must join the fresh flight rather than start a third fetch.
func TestCacheStaleLeaderKeepsFreshFlight(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("v1")); err != nil {
		t.Fatal(err)
	}
	gated := newGatedStore(inner)
	gate1, gate2 := make(chan struct{}), make(chan struct{})
	entered := make(chan struct{})
	gated.gates = []chan struct{}{gate1, gate2, nil}
	gated.entered = entered
	c := New(Config{Underlying: gated})

	waiterJoined := make(chan struct{}, 1)
	testHookWaiterJoined = func() { waiterJoined <- struct{}{} }
	t.Cleanup(func() { testHookWaiterJoined = nil })

	type result struct {
		val  []byte
		err  error
		done chan struct{}
	}
	get := func() *result {
		r := &result{done: make(chan struct{})}
		go func() {
			r.val, r.err = c.Get(ctx, "k")
			close(r.done)
		}()
		return r
	}

	r1 := get()
	<-entered // fetch #1 captured "v1"
	if err := c.Set(ctx, "k", []byte("v2")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	r2 := get()
	select {
	case <-entered: // fetch #2 captured "v2"
	case <-waiterJoined:
		t.Fatal("Get after Set joined the stale flight")
	case <-time.After(5 * time.Second):
		t.Fatal("second Get did not start a fetch")
	}

	close(gate1) // stale fetch #1 completes while fresh fetch #2 is in flight
	<-r1.done

	r3 := get()
	select {
	case <-waiterJoined: // joined fetch #2, as it should
	case <-entered:
		t.Fatal("stale leader dropped the fresh flight: third Get started a new fetch")
	case <-time.After(5 * time.Second):
		t.Fatal("third Get neither joined a flight nor started a fetch")
	}
	close(gate2)
	<-r2.done
	<-r3.done

	if r1.err != nil || string(r1.val) != "v1" {
		t.Fatalf("first Get: got %q err=%v, want %q,nil", r1.val, r1.err, "v1")
	}
	for i, r := range []*result{r2, r3} {
		if r.err != nil || string(r.val) != "v2" {
			t.Fatalf("Get %d: got %q err=%v, want %q,nil", i+2, r.val, r.err, "v2")
		}
	}
	if got := gated.Calls(); got != 2 {
		t.Fatalf("underlying Get calls = %d, want 2", got)
	}
}
