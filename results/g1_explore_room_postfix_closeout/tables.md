### R1 pre-fix, OSMesa, fresh seeds

Baseline: frozen EGL result (results/g1_explore_room_closeout, gain rows)

| kind | seed | base | new | strict | claim | contacts (new) | contacts (base) | sim_s new/base | reason |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 1300 | True | True | True | True | - | - | 25.4/25.4 | found and inspected candidate 2: a box with a handle |
| nominal | 1301 | True | True | True | True | - | - | 23.6/23.6 | found and inspected candidate 2: a box with a handle |
| nominal | 1302 | False | True **flip** | True | True | - | hand_furniture,hand_handle | 106.0/212.0 | found and inspected candidate 5: a box with a handle |
| nominal | 1303 | True | True | True | True | - | - | 55.9/36.3 | found and inspected candidate 2: a box with a handle |
| nominal | 1304 | True | True | False | True | hand_furniture,hand_handle | - | 138.7/26.5 | found and inspected candidate 1: a box with a handle |
| nominal | 1305 | False | True **flip** | False | True | other | hand_furniture | 155.4/183.3 | found and inspected candidate 9: a box with a handle |
| nominal | 1306 | True | True | True | True | - | - | 51.2/47.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1307 | True | True | True | True | - | - | 83.4/85.4 | found and inspected candidate 7: a box with a handle |
| nominal | 1308 | True | True | True | True | - | - | 94.5/88.3 | found and inspected candidate 5: a box with a handle |
| nominal | 1309 | True | True | True | True | - | - | 27.1/27.1 | found and inspected candidate 2: a box with a handle |
| nominal | 1310 | True | True | True | True | - | - | 25.8/25.8 | found and inspected candidate 1: a box with a handle |
| nominal | 1311 | True | True | True | True | - | - | 42.7/48.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1312 | True | False **flip** | False | False | hand_furniture | - | 63.3/56.2 | candidate 4 has a handle, but no reachable viewpoint |
| nominal | 1313 | False | False | False | False | - | - | 0.0/0.0 | scene generation failed: no free start position |
| nominal | 1314 | True | True | True | True | - | - | 66.6/61.5 | found and inspected candidate 1: a box with a handle |
| nominal | 1315 | True | False **flip** | False | False | hand_furniture | - | 164.2/153.0 | candidate 1 has a handle, but no reachable viewpoint |
| nominal | 1316 | False | True **flip** | True | True | - | other | 128.7/193.7 | found and inspected candidate 3: a box with a handle |
| nominal | 1317 | True | True | True | True | - | - | 32.2/53.3 | found and inspected candidate 4: a box with a handle |
| nominal | 1318 | True | True | True | True | - | - | 33.0/33.4 | found and inspected candidate 1: a box with a handle |
| nominal | 1319 | False | False | False | False | - | - | 0.0/0.0 | scene generation failed: no free start position |
| shifted | 2300 | False | False | False | False | hand_furniture,other | hand_furniture | 247.1/254.3 | time budget (240 s) spent |
| shifted | 2301 | True | False **flip** | False | False | hand_furniture | - | 194.6/158.1 | candidate 7 has a handle, but no reachable viewpoint |
| shifted | 2302 | False | False | False | False | hand_furniture,other | hand_furniture | 273.6/243.1 | time budget (240 s) spent |
| shifted | 2303 | True | True | False | True | hand_furniture | hand_furniture | 113.1/110.7 | found and inspected candidate 8: a box with a handle |
| shifted | 2304 | False | True **flip** | False | True | other | - | 145.2/171.4 | found and inspected candidate 7: a box with a handle |
| shifted | 2305 | True | True | True | True | - | - | 77.7/77.5 | found and inspected candidate 6: a box with a handle |
| shifted | 2306 | False | False | False | False | - | hand_furniture,hand_handle | 247.6/251.0 | time budget (240 s) spent |
| shifted | 2307 | True | True | False | True | hand_furniture | hand_furniture,hand_handle | 152.7/236.5 | found and inspected candidate 9: a box with a handle |
| shifted | 2308 | False | False | False | False | hand_furniture,other | hand_furniture | 197.6/231.4 | candidate 6 has a handle, but no reachable viewpoint |
| shifted | 2309 | False | False | False | False | hand_furniture | hand_furniture | 260.6/247.1 | time budget (240 s) spent |
| no_cabinet | 3100 | True | True | False | False | - | - | 123.6/174.0 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3101 | True | True | False | False | hand_furniture | hand_furniture | 180.3/165.8 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3102 | True | True | False | False | - | other | 102.8/228.5 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3103 | True | True | False | False | - | - | 104.0/97.2 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3104 | True | True | False | False | - | other | 109.0/238.4 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3105 | True | True | False | False | hand_furniture | - | 179.4/166.6 | explored the reachable space; 5 candidate object(s) checked, none with a handle |
| no_cabinet | 3106 | True | True | False | False | hand_furniture | hand_furniture | 215.3/180.0 | candidate 2 has a handle, but no reachable viewpoint |
| no_cabinet | 3107 | True | True | False | False | hand_furniture,other | - | 240.2/256.8 | time budget (240 s) spent |
| no_cabinet | 3108 | True | True | False | False | - | - | 90.9/91.1 | explored the reachable space; 2 candidate object(s) checked, none with a handle |
| no_cabinet | 3109 | True | True | False | False | - | hand_furniture | 160.2/212.6 | explored the reachable space; 5 candidate object(s) checked, none with a handle |

### R2 post-fix, EGL, fresh seeds

Baseline: frozen EGL result (results/g1_explore_room_closeout, gain rows)

| kind | seed | base | new | strict | claim | contacts (new) | contacts (base) | sim_s new/base | reason |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 1300 | True | True | True | True | - | - | 25.4/25.4 | found and inspected candidate 2: a box with a handle |
| nominal | 1301 | True | True | True | True | - | - | 23.6/23.6 | found and inspected candidate 2: a box with a handle |
| nominal | 1302 | False | True **flip** | True | True | - | hand_furniture,hand_handle | 134.5/212.0 | found and inspected candidate 7: a box with a handle |
| nominal | 1303 | True | True | True | True | - | - | 36.3/36.3 | found and inspected candidate 2: a box with a handle |
| nominal | 1304 | True | True | True | True | - | - | 26.5/26.5 | found and inspected candidate 1: a box with a handle |
| nominal | 1305 | False | True **flip** | True | True | - | hand_furniture | 116.4/183.3 | found and inspected candidate 7: a box with a handle |
| nominal | 1306 | True | True | True | True | - | - | 47.0/47.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1307 | True | True | True | True | - | - | 85.4/85.4 | found and inspected candidate 6: a box with a handle |
| nominal | 1308 | True | True | True | True | - | - | 88.3/88.3 | found and inspected candidate 5: a box with a handle |
| nominal | 1309 | True | True | True | True | - | - | 27.1/27.1 | found and inspected candidate 2: a box with a handle |
| nominal | 1310 | True | True | True | True | - | - | 25.8/25.8 | found and inspected candidate 1: a box with a handle |
| nominal | 1311 | True | True | True | True | - | - | 48.0/48.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1312 | True | True | True | True | - | - | 56.2/56.2 | found and inspected candidate 5: a box with a handle |
| nominal | 1313 | False | False | False | False | - | - | 0.0/0.0 | scene generation failed: no free start position |
| nominal | 1314 | True | True | True | True | - | - | 61.5/61.5 | found and inspected candidate 1: a box with a handle |
| nominal | 1315 | True | True | True | True | - | - | 154.0/153.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1316 | False | False | False | False | other | other | 193.7/193.7 | candidate 2 has a handle, but no reachable viewpoint |
| nominal | 1317 | True | True | True | True | - | - | 53.3/53.3 | found and inspected candidate 4: a box with a handle |
| nominal | 1318 | True | True | True | True | - | - | 33.4/33.4 | found and inspected candidate 1: a box with a handle |
| nominal | 1319 | False | False | False | False | - | - | 0.0/0.0 | scene generation failed: no free start position |
| shifted | 2300 | False | True **flip** | False | True | hand_furniture | hand_furniture | 108.6/254.3 | found and inspected candidate 8: a box with a handle |
| shifted | 2301 | True | True | True | True | - | - | 158.1/158.1 | found and inspected candidate 1: a box with a handle |
| shifted | 2302 | False | True **flip** | False | True | hand_furniture | hand_furniture | 170.5/243.1 | found and inspected candidate 8: a box with a handle |
| shifted | 2303 | True | True | False | True | hand_furniture | hand_furniture | 111.4/110.7 | found and inspected candidate 8: a box with a handle |
| shifted | 2304 | False | False | False | False | - | - | 171.4/171.4 | candidate 3 has a handle, but no reachable viewpoint |
| shifted | 2305 | True | True | True | True | - | - | 77.5/77.5 | found and inspected candidate 5: a box with a handle |
| shifted | 2306 | False | True **flip** | True | True | - | hand_furniture,hand_handle | 220.0/251.0 | found and inspected candidate 14: a box with a handle |
| shifted | 2307 | True | True | True | True | - | hand_furniture,hand_handle | 167.5/236.5 | found and inspected candidate 10: a box with a handle |
| shifted | 2308 | False | True **flip** | True | True | - | hand_furniture | 88.9/231.4 | found and inspected candidate 6: a box with a handle |
| shifted | 2309 | False | False | False | False | hand_furniture | hand_furniture | 248.4/247.1 | time budget (240 s) spent |
| no_cabinet | 3100 | True | True | False | False | - | - | 174.0/174.0 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3101 | True | True | False | False | hand_furniture | hand_furniture | 165.8/165.8 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3102 | True | True | False | False | other | other | 228.5/228.5 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3103 | True | True | False | False | - | - | 97.2/97.2 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3104 | True | True | False | False | other | other | 238.4/238.4 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3105 | True | True | False | False | - | - | 166.6/166.6 | explored the reachable space; 5 candidate object(s) checked, none with a handle |
| no_cabinet | 3106 | True | True | False | False | hand_furniture | hand_furniture | 180.0/180.0 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3107 | True | True | False | False | - | - | 256.8/256.8 | time budget (240 s) spent |
| no_cabinet | 3108 | True | True | False | False | - | - | 91.1/91.1 | explored the reachable space; 2 candidate object(s) checked, none with a handle |
| no_cabinet | 3109 | True | True | False | False | hand_furniture | hand_furniture | 212.6/212.6 | explored the reachable space; 5 candidate object(s) checked, none with a handle |

### R2 post-fix, EGL, historical seeds

Baseline: A1 (results/g1_explore_room_regression_closeout/A1)

| kind | seed | base | new | strict | claim | contacts (new) | contacts (base) | sim_s new/base | reason |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 1000 | True | True | True | True | - | - | 75.4/75.4 | found and inspected candidate 1: a box with a handle |
| nominal | 1001 | True | True | True | True | - | - | 58.5/58.5 | found and inspected candidate 3: a box with a handle |
| nominal | 1002 | True | True | True | True | - | - | 85.4/85.4 | found and inspected candidate 3: a box with a handle |
| nominal | 1003 | True | True | False | True | other | other | 124.5/124.5 | found and inspected candidate 4: a box with a handle |
| nominal | 1004 | True | True | True | True | - | - | 65.1/65.1 | found and inspected candidate 4: a box with a handle |
| nominal | 1005 | True | True | True | True | - | - | 79.1/79.1 | found and inspected candidate 4: a box with a handle |
| nominal | 1006 | True | True | True | True | - | - | 117.1/117.1 | found and inspected candidate 4: a box with a handle |
| nominal | 1007 | True | True | True | True | - | - | 26.6/26.6 | found and inspected candidate 1: a box with a handle |
| nominal | 1008 | True | True | True | True | - | - | 62.0/62.0 | found and inspected candidate 2: a box with a handle |
| nominal | 1009 | True | True | True | True | - | - | 119.9/119.9 | found and inspected candidate 3: a box with a handle |
| nominal | 1010 | True | True | True | True | - | - | 46.9/46.9 | found and inspected candidate 3: a box with a handle |
| nominal | 1011 | True | True | True | True | - | - | 23.2/23.2 | found and inspected candidate 1: a box with a handle |
| nominal | 1012 | True | True | True | True | - | - | 36.6/36.6 | found and inspected candidate 1: a box with a handle |
| nominal | 1013 | False | True **flip** | False | True | other | hand_furniture,hand_handle,other | 118.2/243.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1014 | True | True | True | True | - | - | 37.3/37.3 | found and inspected candidate 3: a box with a handle |
| nominal | 1015 | True | True | True | True | - | - | 30.1/30.1 | found and inspected candidate 2: a box with a handle |
| nominal | 1016 | True | True | True | True | - | - | 50.7/50.7 | found and inspected candidate 1: a box with a handle |
| nominal | 1017 | True | True | True | True | - | - | 46.8/46.4 | found and inspected candidate 1: a box with a handle |
| nominal | 1018 | True | True | True | True | - | - | 102.0/102.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1019 | True | True | True | True | - | - | 24.7/24.7 | found and inspected candidate 1: a box with a handle |
| shifted | 2000 | False | True **flip** | True | True | - | hand_furniture | 139.4/215.5 | found and inspected candidate 3: a box with a handle |
| shifted | 2001 | True | True | True | True | - | - | 79.9/79.9 | found and inspected candidate 7: a box with a handle |
| shifted | 2002 | True | True | True | True | - | - | 110.0/110.0 | found and inspected candidate 6: a box with a handle |
| shifted | 2003 | True | True | True | True | - | - | 27.6/27.6 | found and inspected candidate 3: a box with a handle |
| shifted | 2004 | False | False | False | False | - | - | 171.3/171.3 | explored the reachable space; 6 candidate object(s) checked, none with a handle |
| shifted | 2005 | True | True | True | True | - | - | 46.8/46.8 | found and inspected candidate 3: a box with a handle |
| shifted | 2006 | True | True | True | True | - | - | 26.3/26.3 | found and inspected candidate 1: a box with a handle |
| shifted | 2007 | True | True | False | True | other | other | 141.8/141.8 | found and inspected candidate 6: a box with a handle |
| shifted | 2008 | False | False | False | False | other | other | 243.9/243.9 | time budget (240 s) spent |
| shifted | 2009 | False | False | False | False | - | - | 249.6/249.6 | time budget (240 s) spent |
| no_cabinet | 3000 | True | True | False | False | other | other | 132.7/132.7 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3001 | True | True | False | False | - | - | 130.1/130.1 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3002 | True | True | False | False | other | other | 205.5/205.5 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3003 | True | True | False | False | - | - | 97.7/97.7 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3004 | True | True | False | False | - | - | 126.3/126.3 | explored the reachable space; 2 candidate object(s) checked, none with a handle |
| no_cabinet | 3005 | True | True | False | False | - | - | 118.8/118.8 | explored the reachable space; 4 candidate object(s) checked, none with a handle |

### R3 post-fix, OSMesa, historical seeds

Baseline: historical OSMesa record (results/g1_explore_room, gain rows)

| kind | seed | base | new | strict | claim | contacts (new) | contacts (base) | sim_s new/base | reason |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 1000 | True | True | True | True | - | - | 74.5/74.0 | found and inspected candidate 1: a box with a handle |
| nominal | 1001 | True | True | True | True | - | - | 45.5/64.5 | found and inspected candidate 3: a box with a handle |
| nominal | 1002 | True | True | True | True | - | - | 76.3/78.2 | found and inspected candidate 3: a box with a handle |
| nominal | 1003 | True | True | False | True | other | - | 156.5/156.2 | found and inspected candidate 4: a box with a handle |
| nominal | 1004 | True | True | True | True | - | - | 64.7/62.7 | found and inspected candidate 4: a box with a handle |
| nominal | 1005 | True | True | True | True | - | - | 79.1/79.1 | found and inspected candidate 5: a box with a handle |
| nominal | 1006 | True | True | False | True | other | - | 142.1/142.1 | found and inspected candidate 4: a box with a handle |
| nominal | 1007 | True | True | True | True | - | - | 26.6/26.6 | found and inspected candidate 1: a box with a handle |
| nominal | 1008 | True | True | True | True | - | - | 61.7/89.0 | found and inspected candidate 2: a box with a handle |
| nominal | 1009 | True | True | True | True | - | - | 112.7/112.7 | found and inspected candidate 3: a box with a handle |
| nominal | 1010 | True | True | True | True | - | - | 36.1/40.9 | found and inspected candidate 3: a box with a handle |
| nominal | 1011 | True | True | True | True | - | - | 23.2/26.6 | found and inspected candidate 1: a box with a handle |
| nominal | 1012 | True | True | True | True | - | - | 37.2/54.7 | found and inspected candidate 1: a box with a handle |
| nominal | 1013 | True | True | False | True | other | - | 118.2/116.0 | found and inspected candidate 3: a box with a handle |
| nominal | 1014 | True | True | True | True | - | - | 38.2/47.9 | found and inspected candidate 3: a box with a handle |
| nominal | 1015 | True | True | True | True | - | - | 30.0/33.1 | found and inspected candidate 2: a box with a handle |
| nominal | 1016 | True | True | True | True | - | - | 74.5/71.6 | found and inspected candidate 1: a box with a handle |
| nominal | 1017 | True | True | True | True | - | - | 34.2/34.2 | found and inspected candidate 1: a box with a handle |
| nominal | 1018 | True | True | True | True | - | - | 21.8/26.6 | found and inspected candidate 3: a box with a handle |
| nominal | 1019 | True | True | True | True | - | - | 24.7/26.4 | found and inspected candidate 1: a box with a handle |
| shifted | 2000 | True | True | False | True | other | - | 239.8/239.4 | found and inspected candidate 17: a box with a handle |
| shifted | 2001 | True | True | True | True | - | - | 63.0/64.4 | found and inspected candidate 7: a box with a handle |
| shifted | 2002 | True | True | True | True | - | - | 166.3/168.5 | found and inspected candidate 6: a box with a handle |
| shifted | 2003 | True | True | True | True | - | - | 23.6/25.1 | found and inspected candidate 3: a box with a handle |
| shifted | 2004 | True | True | True | True | - | - | 70.8/70.8 | found and inspected candidate 1: a box with a handle |
| shifted | 2005 | True | True | True | True | - | - | 45.5/43.7 | found and inspected candidate 3: a box with a handle |
| shifted | 2006 | True | True | True | True | - | - | 28.8/28.8 | found and inspected candidate 1: a box with a handle |
| shifted | 2007 | True | True | False | True | other | - | 122.5/122.5 | found and inspected candidate 4: a box with a handle |
| shifted | 2008 | False | False | False | False | other | - | 245.3/245.3 | time budget (240 s) spent |
| shifted | 2009 | False | False | False | False | hand_furniture | - | 293.7/253.3 | time budget (240 s) spent |
| no_cabinet | 3000 | True | True | False | False | other | - | 137.6/137.6 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3001 | True | True | False | False | - | - | 148.5/172.3 | candidate 2 has a handle, but no reachable viewpoint |
| no_cabinet | 3002 | True | True | False | False | other | - | 210.6/210.6 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
| no_cabinet | 3003 | True | True | False | False | - | - | 114.9/114.9 | explored the reachable space; 3 candidate object(s) checked, none with a handle |
| no_cabinet | 3004 | True | True | False | False | - | - | 125.8/125.8 | explored the reachable space; 2 candidate object(s) checked, none with a handle |
| no_cabinet | 3005 | True | True | False | False | hand_furniture | - | 145.8/145.8 | explored the reachable space; 4 candidate object(s) checked, none with a handle |
