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
            if (l is SimulationGameEventListener)
            {
                SimulationGameEventListener l1 = l as SimulationGameEventListener;
                l1.OnServerConnectedAttemptEventRaised(e);
            }
            else if (l is MainMenuGameEventListener)
            {
                MainMenuGameEventListener l1 = l as MainMenuGameEventListener;
                l1.OnServerConnectedAttemptEventRaised(e);
            }
        }
    }
}
