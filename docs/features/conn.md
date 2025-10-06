## Connection Log
NOTE: for now, due to implementation challenges, scaling by service is NOT performed.
#### New Fields:
- [x] byte_ratio: orig_bytes/resp_bytes, standard scaled
- [x] packet_ratio: orig_pkts/resp_pkts, standard scaled
- [x] orig_packet_size: orig_bytes/orig_pkts, standard scaled
- [x] resp_packet_size: resp_bytes/resp_pkts, standard scaled
- [x] in_tunnel: Is the connection tunneled or exposed?
- [x] good_history: Is the conn_state normal for this protocol?

#### Modified Fields:
- [x] proto: OHE
- [x] duration: standard scaled BY SERVICE (dns will all be scaled the same, as will http, etc.)
- [x] local_orig: binary
- [x] local_resp: binary
- [x] orig_pkts: standard scaled BY SERVICE
- [x] resp_pkts: standard scaled BY SERVICE

#### Removed Fields:
- [x] All IDs (IPs/ports) (this will be captured in the structural model)
- [x] service (this will be captured in the sequence model)
- [x] orig_bytes (just focusing on packets)
- [x] resp_bytes
- [x] orig_ip_bytes (not useful)
- [x] resp_ip_bytes
- [x] tunnel_parents (Knowing their UIDs might help, but likely not. Even if we know them, we can't parse the data here)
- [x] orig_l2_addr (not dealing with layer 2 info as this would be too complicated given the size of our data)
- [x]  resp_l2_addr
- [x] vlan (this really only matters if the VLANs shouldn't be communicating, which is captured by the new variable)
- [x] inner_vlan
- [x] conn_state (only really care if it has a weird conn_state or history)
- [x] history
- [x] resp_cc (this might not always be available)
- [x] community_id (this is some kind of hash used for joining, not needed and offers no info)