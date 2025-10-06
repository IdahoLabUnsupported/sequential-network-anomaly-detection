## Kerberos Log
It is tough to tell from network alone if the kerberos authentication and ticket granting servers are compromised.
That said, we can tell if users might be. It should be very rare that the authenticity of a kerberos message cannot
be verified. I ignore failed PREAUTH because it is possible passwords are not typed properly. Without knowing 
how many attempts there were, it's tough to work with this.

Things that are pretty suspicious would be unkown principals and modified messages. Cryptographic checks should not
normally fail under normal circumstances and could be a sign of a replay attack or issues with the network. Unknown
principals are also odd, as unauthorized users shouldn't be trying to gain access in the first place.

#### New Fields:
- [x] sus_failure (was the failure due to an unknown principal or a failed cryptographic check)