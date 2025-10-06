## DNS Log
#### New Fields
- [x] safe_port (53, 5355, 5353)
- [x] weird_resp_code (not in [0, 1, 3])
- [x] qclass_IN (qclass == 1)
- [x] resolved (answers not null)
###### These would be very nice to implement at some point and are probably the most important
See the geoip2 library for the country
- [ ] bad_query (is the query unsafe)
- [ ] bad_answer (is the answer unsafe)
- [ ] foreign_answer (is the response IP not from the United States)


#### Modified Fields
- [x] AA (binary)
- [x] TC (binary)

#### Removed Fields
- [x] RA (these aren't important for malicious detection)
- [x] RD
- [x] Z
- [x] rcode_name (covered by weird_resp_code)
- [x] rcode
- [x] qclass (encompassed via qclass_IN)
- [x] qclass_name
- [x] TTLs (corrupted)
- [x] All ID + Port info (safe_port has this covered)
- [x] proto (Handled by the conn log)
- [x] trans_id (too many to count)
- [x] query (too many options to use raw)
- [x] answers
- [x] rtt (unreliable and almost always NaN)