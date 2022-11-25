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
            if (l is SimulationGameEventListener)
            {
                SimulationGameEventListener l1 = l as SimulationGameEventListener;
                l1.OnServerConnectingEventRaised(ip, port);
            }
            else if (l is MainMenuGameEventListener)
            {
                MainMenuGameEventListener l1 = l as MainMenuGameEventListener;
                l1.OnServerConnectingEventRaised(ip, port);
            }
        }
    }
}
