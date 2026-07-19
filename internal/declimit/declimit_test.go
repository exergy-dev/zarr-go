package declimit

import (
	"math"
	"testing"
)

func TestCheckDeclaredSize(t *testing.T) {
	orig := Max()
	defer SetMax(orig)

	SetMax(1024)
	tests := []struct {
		name    string
		n       int64
		wantErr bool
	}{
		{"negative", -1, true},
		{"zero", 0, false},
		{"at cap", 1024, false},
		{"over cap", 1025, true},
		{"way over cap", 1 << 40, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckDeclaredSize(tt.n)
			if (err != nil) != tt.wantErr {
				t.Fatalf("CheckDeclaredSize(%d) error = %v, wantErr %v", tt.n, err, tt.wantErr)
			}
		})
	}
}

// TestCheckDeclaredSizeCapDisabled verifies that SetMax(0) disables the
// byte-count cap while the negative/MaxInt sanity checks still apply.
func TestCheckDeclaredSizeCapDisabled(t *testing.T) {
	orig := Max()
	defer SetMax(orig)

	SetMax(0)
	if err := CheckDeclaredSize(1 << 40); err != nil {
		t.Fatalf("cap disabled: CheckDeclaredSize(1<<40) = %v, want nil", err)
	}
	if err := CheckDeclaredSize(-1); err == nil {
		t.Fatal("cap disabled: CheckDeclaredSize(-1) = nil, want error (negative is always invalid)")
	}
}

// TestCheckDeclaredSizeMaxIntGuard documents the platform-int overflow guard:
// on 32-bit builds (GOARCH=386/arm), a declared size that fits in int64 but
// exceeds the platform's int range must be rejected before it reaches a
// `make([]byte, int(n))` call, which would otherwise panic or wrap negative.
// On 64-bit builds this branch is unreachable (int64's own range equals
// math.MaxInt), so this test only exercises the boundary that IS reachable
// here; the 32-bit path is covered by the GOARCH=386 CI build.
func TestCheckDeclaredSizeMaxIntGuard(t *testing.T) {
	orig := Max()
	defer SetMax(orig)
	SetMax(0) // isolate the MaxInt guard from the byte-cap guard

	if err := CheckDeclaredSize(math.MaxInt64); err != nil {
		t.Fatalf("CheckDeclaredSize(MaxInt64) = %v, want nil on this platform (MaxInt64 == math.MaxInt)", err)
	}
}
