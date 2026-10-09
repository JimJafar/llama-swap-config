#!/bin/sh
# SSD throughput with fio, O_DIRECT (no page cache), ~80 s per drive.
# Usage: bench/disk.sh DIR [DIR...]   e.g. bench/disk.sh /home/jim /mnt/models
# Each DIR gets a temporary 4 GiB test file, deleted afterwards.
set -eu
for dir in "$@"; do
    f="$dir/.fio-bench.tmp"
    dev=$(findmnt -no SOURCE -T "$dir")
    echo "== $dir ($dev)"
    common="--filename=$f --size=4G --direct=1 --ioengine=io_uring --group_reporting --output-format=json"
    for job in "seqread  --rw=read      --bs=1M --iodepth=32 --numjobs=1 --runtime=20" \
               "seqwrite --rw=write     --bs=1M --iodepth=32 --numjobs=1 --runtime=20" \
               "rand4k-q32 --rw=randread --bs=4k --iodepth=32 --numjobs=4 --runtime=15" \
               "rand4k-q1  --rw=randread --bs=4k --iodepth=1  --numjobs=1 --runtime=15"; do
        name=${job%% *}
        # shellcheck disable=SC2086
        fio --name="$name" ${job#* } --time_based $common | python3 -c '
import json, sys
j = json.load(sys.stdin)["jobs"][0]
name = sys.argv[1]
side = j["write"] if "write" in name else j["read"]
bw = side["bw_bytes"] / 1e6
lat = side["clat_ns"]["mean"] / 1000
iops = side["iops"]
print(f"  {name:11s} {bw:8.0f} MB/s  {iops:9.0f} IOPS  mean latency {lat:7.1f} us")' "$name"
    done
    rm -f "$f"
done
