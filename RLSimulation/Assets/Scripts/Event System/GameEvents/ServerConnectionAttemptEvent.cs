using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ServerConnectionAttemptEvent : GameEvent
{
    public void Raise(Exception e)
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnServerConnectedAttemptEventRaised(e);
        }
    }
}
