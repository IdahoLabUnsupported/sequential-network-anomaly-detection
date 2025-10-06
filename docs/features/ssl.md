## SSL Log
Most of the time, SSL will be tough to detect anomalies with. All traffic is encrypted. What would be weird or 
potentially harmful is if we are running a version of TLS that is out of date and using a cipher with known
security flaws. Most things should be elliptic curve based or RSA with a substantial length key (3072 I think?).

#### New Fields
- [x] safe_port (443)
- [x] odd_version (are you running an SSL version that is different from 95% of your peers)
- [x] elliptic_curve (Is ECDH(E) with Galois mode used? It should almost always be on modern systems)
- [ ] good_history (NEED TO FIND A HISTORY DOCUMENTATION)

#### Modified Fields
- [x] resumed (binary)
- [x] established (binary)

#### Removed Fields
All other fields not lsited above are removed (with the exception of ts and uid)
- [x] cipher (almost everything is elliptic curve based anyway)
- [x] curve
- [x] server_name (too many to count)
- [x] ja3(s) (can't do anything with these quickly. Really only useful if we know the connection was malicious)
