.PHONY: test race vet bench lint tidy fixtures

# The root module plus the nested cloud store modules. `go test ./...` stops
# at a module boundary, so every target that should cover the whole repo
# loops over these.
MODULES := . store/s3 store/gcs store/azblob

test:
	@set -e; for m in $(MODULES); do echo "== $$m"; (cd $$m && go test ./...); done

race:
	@set -e; for m in $(MODULES); do echo "== $$m"; (cd $$m && go test -race ./...); done

vet:
	@set -e; for m in $(MODULES); do echo "== $$m"; (cd $$m && go vet ./...); done

bench:
	go test -bench=. -benchmem -run=^$$ ./...

lint:
	@set -e; for m in $(MODULES); do echo "== $$m"; (cd $$m && golangci-lint run); done

tidy:
	@set -e; for m in $(MODULES); do echo "== $$m"; (cd $$m && go mod tidy); done

fixtures:
	cd testdata/gen && uv run --python 3.11 --with-requirements requirements-v3.txt generate.py
	cd testdata/gen && uv run --python 3.11 --with-requirements requirements-v2.txt generate_v2.py
