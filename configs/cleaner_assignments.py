# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from data.log_cleaners import (
    ConnCleaner,
    HTTPCleaner,
    FilesCleaner,
    KerberosCleaner,
    SSLCleaner,
    NoCleaner
)

"""
This config file determines which log is handled by which cleaner. These cleaners are written in data.log_cleaners.

This config file is used every time a ZeekCleaner is invoked in order to make assignments. GeneralCleaner is repeated
frequently and is just an all purpose cleaner that should (very naively) handle any kind of input. I do not recommend
using a GeneralCleaner for production. It is entirely designed to be a code testing placeholder and should be replaced.

These name are infered from the name of the files they are read from. For example, if a file is called conn.log, it will
be handled as 'conn' (even if it is erroneously named and contains contents of another log).
"""

transforms = {
    "conn": ConnCleaner(),
    "dhcp": NoCleaner(),
    "dns": NoCleaner(),
    "http": HTTPCleaner(),
    "dpd": NoCleaner(),
    "files": FilesCleaner(),
    "ftp": NoCleaner(),
    "irc": NoCleaner(),
    "kerberos": KerberosCleaner(),
    "mysql": NoCleaner(),
    "radius": NoCleaner(),
    "sip": NoCleaner(),
    "software": NoCleaner(),
    "ssh": NoCleaner(),
    "ssl": SSLCleaner(),
    "syslog": NoCleaner(),
    "weird": NoCleaner(),
    "x509": NoCleaner(),
    "dce_rpc": NoCleaner(),
    "ntlm": NoCleaner(),
    "rdp": NoCleaner(),
    "smb_files": NoCleaner(),
    "smb_mapping": NoCleaner()
}
