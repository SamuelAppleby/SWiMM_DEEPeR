using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ResetEpisodeEvent : GameEvent
{
    public void Raise()
    {
        foreach (GameEventListener l in listeners)
        {
            if (l is SimulationGameEventListener)
            {
                SimulationGameEventListener l1 = l as SimulationGameEventListener;
                l1.OnEpisodeResetEventRaised();
            }
            else if (l is MainMenuGameEventListener)
            {
                MainMenuGameEventListener l1 = l as MainMenuGameEventListener;
                l1.OnEpisodeResetEventRaised();
            }
        }
    }
}
