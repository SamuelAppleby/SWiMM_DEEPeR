import argparse

from gym_underwater.enums import Protocol

parser = argparse.ArgumentParser()
parser.add_argument('--host', help='Override the host for network (with port)', default='127.0.0.1:60260', type=str)
parser.add_argument('-tcp', help='Enable tcp', action='store_true')
parser.add_argument('-udp', help='Enable udp', action='store_true')
args = parser.parse_args()

if args.udp and not args.tcp:
    args.protocol = Protocol.UDP
else:
    args.protocol = Protocol.TCP
