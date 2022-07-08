#!/bin/bash
for file in BDD100K/train/*.json
do
	gpsbabel -i geojson -f "$file" -o gpx -F "${file/json/gpx}"
done
