## Files Log
Files can tell us a little bit about the kind of data being sent and its origin.
All html and xml files are just discarded since it's likely that the browser is just serving them.
These can be handled by the http log instead.

#### New Fields
- [x] sus_ext (are we downloading an executable or something)

#### Modified Fields
- [x] seen_bytes
- [x] total_bytes
- [x] is_orig
- [x] entropy
- [x] extracted_size

#### Removed Fields
All other fields not listeds above are removed (with the exception of ts and uid)