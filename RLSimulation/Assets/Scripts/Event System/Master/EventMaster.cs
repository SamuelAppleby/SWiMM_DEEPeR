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

    [SerializeField]
    public AIGroupsCompleteEvent ai_groups_complete_event;

    [SerializeField]
    public ServerConfigReceivedEvent server_config_received_event;

    [SerializeField]
    public ServerConnectingEvent server_connecting_event;

    [SerializeField]
    public ServerConnectionAttemptEvent server_connection_attempt_event;

    [SerializeField]
    public ObservationSentEvent observation_sent_event;

    [SerializeField]
    public ServerAwaitingTrainingEvent server_awaiting_training_event;

    protected override void Awake()
    {
        base.Awake();
        GetComponent<GameEventListener>().enabled = true;
    }
}
