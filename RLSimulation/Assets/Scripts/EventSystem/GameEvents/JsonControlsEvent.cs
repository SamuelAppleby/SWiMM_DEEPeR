using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Server;

[CreateAssetMenu]
public class JsonControlsEvent : GameEvent
{
    public void Raise(JsonMessage<JsonControls> param)
    {
        foreach (GameEventListener l in listeners)
        {
            if (l is SimulationGameEventListener)
            {
                SimulationGameEventListener l1 = l as SimulationGameEventListener;
                l1.OnJsonControlEventRaised(param);
            }
            else if (l is PlayerGameEventListener)
            {
                PlayerGameEventListener l1 = l as PlayerGameEventListener;
                l1.OnJsonControlEventRaised(param);
            }
        }
    }
}