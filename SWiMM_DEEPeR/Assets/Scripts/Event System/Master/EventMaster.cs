using UnityEngine;

public class EventMaster : MonoBehaviour
{
    [SerializeField]
    public ROVInitialisedEvent rov_initialised_event;

    [SerializeField]
    public AIGroupsCompleteEvent ai_groups_complete_event;

    [SerializeField]
    public ServerConnectingEvent server_connecting_event;

    [SerializeField]
    public ServerConnectionAttemptEvent server_connection_attempt_event;

    [SerializeField]
    public ObservationSentEvent observation_sent_event;

    [SerializeField]
    public SceneChangedEvent scene_changed_event;

    [SerializeField]
    public ServerEvent server_event;

    [SerializeField]
    public SentEvent sent_event;

    protected void Awake()
    {
        GetComponent<GameEventListener>().enabled = true;
    }
}
