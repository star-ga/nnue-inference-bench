PYTHON ?= python3
CMAKE ?= cmake

.PHONY: help smoke cpp cuda mind bench validate docs docs-check clean

help:
	@echo "make smoke       reference smoke test (pinned checksum)"
	@echo "make cpp         build C++ scalar + AVX2 (native-cpp/build)"
	@echo "make cuda        build CUDA kernels (native-cuda/build)"
	@echo "make mind        build the MIND kernel (mind/build)"
	@echo "make bench       run every available path, write build/work/results.json"
	@echo "make validate    check build/work/results.json against the pinned checksum"
	@echo "make docs        regenerate the README table from results/latest.json"
	@echo "make docs-check  fail if the README table differs from results/latest.json"

smoke:
	$(PYTHON) tools/smoke_cpu.py

cpp:
	$(CMAKE) -S native-cpp -B native-cpp/build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build native-cpp/build --config Release

cuda:
	$(CMAKE) -S native-cuda -B native-cuda/build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build native-cuda/build --config Release

mind:
	$(PYTHON) tools/infer_mind.py

bench:
	$(PYTHON) tools/run_benchmark.py -o build/work/results.json

validate:
	$(PYTHON) tools/validate_results.py build/work/results.json

docs:
	$(PYTHON) tools/update_docs.py

docs-check:
	$(PYTHON) tools/update_docs.py --check

clean:
	rm -rf build native-cpp/build native-cuda/build mind/build
