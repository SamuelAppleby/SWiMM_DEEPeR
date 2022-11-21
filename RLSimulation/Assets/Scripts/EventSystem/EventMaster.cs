using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EventMaster : Singleton<EventMaster>
{
    [SerializeField]
    public ROVInitialisedEvent rov_initialised_event;

    [SerializeField]
    public EndSimulationEvent end_simulation_event;

    [SerializeField]
    public ResetEpisodeEvent reset_episode_event;

    [SerializeField]
    public JsonControlsEvent json_control_event;

    protected override void Awake()
    {
        base.Awake();
        GetComponent<SimulationGameEventListener>().enabled = true;
    }
}
