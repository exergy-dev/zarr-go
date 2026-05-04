.PHONY: test race bench lint tidy fixtures

test:
	go test ./...

race:
	go test -race ./...

bench:
	go test -bench=. -benchmem -run=^$$ ./...

lint:
	golangci-lint run

tidy:
	go mod tidy

fixtures:
	cd testdata/gen && python generate.py
