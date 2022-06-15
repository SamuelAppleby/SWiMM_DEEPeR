using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

[System.Serializable]
public class AIObjects
{
    public GameObject object_prefab { get { return m_object_prefb; } }
    public int max_ai { get { return m_max_ai; } }
    public int max_spawn_amount { get { return m_max_spawn_amount; } }
    public bool randomize_stats { get { return m_randomize_stats; } }
    public bool enable_spawner { get { return m_enable_spawner; } }
    public bool random_movement { get { return m_random_movement; } }

    [Header("AI Group Stats")]
    [SerializeField]
    private GameObject m_object_prefb;
    [SerializeField]
    [Range(0f, 20f)]
    private int m_max_ai;
    [SerializeField]
    [Range(0f, 10f)]
    private int m_max_spawn_amount;
    [SerializeField]
    private bool m_random_movement;


    [Header("Main Settings")]
    [SerializeField]
    private bool m_randomize_stats;
    [SerializeField]
    private bool m_enable_spawner;

    public AIObjects(GameObject prefab, int max_ai, int spawn_rate, int spawn_amount, bool random_movement, bool randomize_stats)
    {
        m_object_prefb = prefab;
        m_max_ai = max_ai;
        m_max_spawn_amount = spawn_amount;
        m_random_movement = random_movement;
        m_randomize_stats = randomize_stats;
    }

    public void Randomise()
    {
        m_max_ai = Random.Range(1, 20);
        m_max_spawn_amount = Random.Range(1, 10);
    }
}

public class FishSpawner : MonoBehaviour
{
    public Transform[] waypoints;

    public float spawn_timer { get { return m_spawn_timer; } }
    public Vector3 spawn_area { get { return m_spawn_area; } }
    public int spawn_container_ratio { get { return m_spawn_container_ratio; } }

    [Header("Global Stats")]
    [Range(0f, 600f)]
    [SerializeField]
    private float m_spawn_timer;
    [SerializeField]
    private Color m_spawn_color = new Color(1f, 0f, 0f, 0.3f);
    [SerializeField]
    private Collider m_water_collider;
    [SerializeField]
    [Range(0f, 100f)]
    private int m_spawn_container_ratio;

    [Header("AI Group Settings")]
    public AIObjects[] ai_objects = new AIObjects[1];

    private Vector3 m_spawn_area;

    // Start is called before the first frame update
    void Start()
    {
        CreateSpawnableArea();
        GetWaypoints();
        InitialiseGroups();
        InvokeRepeating("SpawnNPC", 0.5f, spawn_timer);
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void CreateSpawnableArea()
    {
        BoxCollider col = m_water_collider as BoxCollider;
        if (col)
        {
            m_spawn_area = new Vector3((m_spawn_container_ratio / 10) * col.size.x, (m_spawn_container_ratio / 100) * col.size.y, (m_spawn_container_ratio / 10) * col.size.z);
        }
    }

    private void GetWaypoints()
    {
        waypoints = transform.GetComponentsInChildren<Transform>().Where(c => c.gameObject.tag == "Waypoint").ToArray();
    }

    private void InitialiseGroups()
    {
        foreach (AIObjects obj in ai_objects)
        {
            if(obj.object_prefab != null)
            {
                if (obj.randomize_stats)
                {
                    obj.Randomise();
                }

                GameObject m_ai_group_spawn = new GameObject(obj.object_prefab.name);
                m_ai_group_spawn.transform.parent = this.gameObject.transform;
            }
        }
    }

    private void SpawnNPC()
    {
        foreach (AIObjects obj in ai_objects)
        {
            if (obj.object_prefab != null && obj.enable_spawner)
            {
                GameObject temp_group = GameObject.Find(obj.object_prefab.name);
                if (temp_group != null && temp_group.GetComponentInChildren<Transform>().childCount < obj.max_ai)
                {
                    for (int i = 0; i < Random.Range(0, obj.max_spawn_amount + 1); ++i)
                    {
                        Quaternion random_rotation = Quaternion.Euler(Random.Range(-20, 20), Random.Range(0, 360), 0);
                        Vector3 random_position = GetRandomPosition();
                        if(IsValidLocation(random_position, 30))
                        {
                            GameObject temp_spawn = Instantiate(obj.object_prefab, random_position, random_rotation);
                            temp_spawn.transform.parent = temp_group.transform;
                            temp_spawn.AddComponent<FishMovement>();
                            temp_spawn.GetComponent<FishMovement>().random_movement = obj.random_movement;

                            SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().target_transforms.Add(temp_spawn.transform);
                        }
                    }
                }
            }
        }
    }

    public Vector3 GetRandomPosition()
    {
        Vector3 random_position = new Vector3(Random.Range(-spawn_area.x, spawn_area.x),
            Random.Range(-spawn_area.y, spawn_area.y),
            Random.Range(-spawn_area.z, spawn_area.z)) * .5f;

        return random_position;
    }

    public bool IsValidLocation(Vector3 pos, float radius)
    {
        return Physics.CheckSphere(pos, radius);
    }

    public Vector3 RandomWaypoint()
    {
        return waypoints[Random.Range(0, (waypoints.Count() - 1))].position;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = m_spawn_color;
        Gizmos.DrawCube(transform.position, spawn_area);
    }
}
