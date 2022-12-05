using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ServerAwaitingTrainingEvent : GameEvent
{
    public void Raise()
    {
        foreach (MainMenuGameEventListener l in listeners)
        {
            l.OnServerAwaitingTrainingEventRaised();
        }
    }
}
