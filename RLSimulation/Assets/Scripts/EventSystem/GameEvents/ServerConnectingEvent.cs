using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ServerConnectingEvent : GameEvent
{
    public void Raise(string ip, int port)
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnServerConnectingEventRaised(ip, port);
        }
    }
}
