#!/bin/bash
grep "running simulation..." Log/job.out.*.* | sort -n -k3 -r
tar -czvf Log.tar.gz Log
