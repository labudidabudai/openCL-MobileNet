main: create_dir
	make main -C src

all: create_dir
	make all -C src

devices: create_dir
	make devices -C src

with_debug: create_dir
	make with_debug -C src

compare_outputs: create_dir
	make compare_outputs -C src

create_dir:
	mkdir -p bin

clear:
	rm -f bin/*

test: main compare_outputs
	cd bin && ./opencl_mobilenet ../test/images_list.txt test_output.txt
	bin/compare_outputs bin/test_output.txt test/etalon_output.txt
	rm bin/test_output.txt
