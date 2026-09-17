"""Lets ``python -m blast_radius`` run the command line."""

import sys

from blast_radius import cli

if __name__ == "__main__":
  sys.exit(cli.main())
