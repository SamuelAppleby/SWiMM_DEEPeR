using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ROVInitialisedEvent : GameEvent
{
    public void Raise(GameObject obj)
    {
        foreach (GameEventListener l in listeners)
        {
            if (l is SimulationGameEventListener)
            {
                SimulationGameEventListener l1 = l as SimulationGameEventListener;
                l1.OnROVInitalisedEventRaised(obj);
            }
            else if (l is FishSpawnerGameEventListener)
            {
                FishSpawnerGameEventListener l1 = l as FishSpawnerGameEventListener;
                l1.OnROVInitalisedEventRaised(obj);
            }
        }
    }
}
