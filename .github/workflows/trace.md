Comparing main branch to debug branch
Initialize container went down from   38s (before merge) down to 9s
Install required packages:            8s     -> 1s (hit) + 6s + 0s
                                             -> 0s (miss) + 8s + ?
Update conda and install conda build: 32s    -> 0s (miss) + 37s + 21s
                                             -> 1m 11s (hit) + 42s +
Install python:                       6s     -> 7s
Build conda package:                  8m 32s -> 1s + 9m 28s + 0s (hit)

Cache not found for input keys: conda-cache-amd64-3.10

Cache Size: ~1 MB (539303 B)
Cache restored from key: sccache-amd64-3.10

Cache hit occurred on the primary key sccache-amd64-3.10, not saving cache.

Cache saved with key: conda-cache-amd64-3.10

echo
