#!/bin/bash

file="""$1"".txt"
touch "$file"
exec 1>"$file"
exec 2>/dev/null
python3 generate_synthetic_raw.py
python3 raw_to_splade.py
python3 seismic/scripts/convert_json_to_inner_format.py --document-path my_splade_docs.jsonl --query-path my_splade_queries.jsonl --output-dir synthetic_data_out --input-format msmarco
./seismic/target/release/build_inverted_index -i synthetic_data_out/data/documents.bin -o synthetic_data_out/data/inv_index.bin
./seismic/target/release/generate_groundtrouth -i synthetic_data_out/data/documents.bin -q synthetic_data_out/data/queries.bin -o synthetic_data_out/data/ground.tsv
./seismic/target/release/perf_inverted_index -i synthetic_data_out/data/inv_index.bin.index.seismic -q synthetic_data_out/data/queries.bin -o synthetic_data_out/data/res_perf.tsv
python3 seismic/scripts/accuracy.py synthetic_data_out/data/ground.tsv synthetic_data_out/data/res_perf.tsv