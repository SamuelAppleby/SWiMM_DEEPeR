using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;

/*
 * TCP communication utility class.
 * I don't know which are the specs for the actual Rover, but I suppose that it would be
 * better to have a UDP communication, as packet loss is not a major problem (take only the most recent one,
 * and discard the others, by using Lamport's clocks).
 */

