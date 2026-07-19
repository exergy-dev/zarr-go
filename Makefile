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
	cd testdata/gen && uv run --python 3.11 --with-requirements requirements-v3.txt generate.py
	cd testdata/gen && uv run --python 3.11 --with-requirements requirements-v2.txt generate_v2.py
