using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FishMovement : MonoBehaviour
{
    public bool random_movement = false;
    private FishSpawner m_ai_manager;
    private Vector3 m_waypoint;
    private Vector3 m_last_waypoint = new Vector3(0f, 0f, 0f);
    private Animation m_animation;
    private float m_speed;
    private Tuple<float, float> m_mix_max_speed = new Tuple<float, float>(1, 7);

    private Collider m_collider;

    // Start is called before the first frame update
    void Start()
    {
        m_ai_manager = transform.parent.GetComponentInParent<FishSpawner>();
        m_animation = GetComponent<Animation>();
        SetUpNPC();
        FindNewTarget();
    }

    // Update is called once per frame
    private void Update()
    {
        ResolveCollisions();

        if((m_waypoint - transform.position).magnitude < 10)
        {
            FindNewTarget();
        }
    }

    private void FixedUpdate()
    {
        float turn_speed = m_speed * UnityEngine.Random.Range(1f, 3f);

        Vector3 look_at = (transform.position - m_waypoint);
        transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(look_at), Time.deltaTime * turn_speed);
        transform.position = Vector3.MoveTowards(transform.position, m_waypoint, m_speed * Time.deltaTime);
    }

    private Vector3 GetWaypoint()
    {
        return random_movement ? m_ai_manager.GetRandomPosition() : m_ai_manager.RandomWaypoint();
    }

    private void FindNewTarget()
    {
        m_last_waypoint = m_waypoint;
        m_waypoint = GetWaypoint();
        m_speed = UnityEngine.Random.Range(m_mix_max_speed.Item1, m_mix_max_speed.Item2);
    }

    private void SetUpNPC()
    {
        float m_scale = UnityEngine.Random.Range(0f, 4f);
        transform.localScale *= m_scale;

        if (transform.GetComponent<Collider>() != null && transform.GetComponentInChildren<Collider>().enabled)
        {
            m_collider = transform.GetComponent<Collider>();
        }
        else if (transform.GetComponentInChildren<Collider>() != null && transform.GetComponentInChildren<Collider>().enabled)
        {
            m_collider = transform.GetComponentInChildren<Collider>();
        }
    }

    private void ResolveCollisions()
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, 10.0f))
        {
            if (hit.collider == m_collider)
            {
                return;
            }
            else if (hit.collider.tag == "Waypoint" || hit.collider.tag == "Terrain" || UnityEngine.Random.Range(1, 100) < 40)
            {
                FindNewTarget();
            }
        }
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Vector3 forward = transform.TransformDirection(Vector3.forward) * 10;
        Debug.DrawRay(transform.position, forward, Color.green);
    }
}
