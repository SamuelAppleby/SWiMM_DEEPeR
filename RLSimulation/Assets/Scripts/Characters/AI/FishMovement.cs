using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FishMovement : MonoBehaviour
{
    public bool random_movement = false;
    public FishSpawner ai_manager;
    private Vector3 m_waypoint;
    private Vector3 m_last_waypoint = new Vector3(0f, 0f, 0f);
    private Animation m_animation;
    private AudioSource call;
    private float call_timer = 10f;
    private float m_speed;
    private Tuple<float, float> m_mix_max_speed = new Tuple<float, float>(1, 7);

    // For axis fixing import from fbx, 3dsmax etc
    public Vector3 rotation_offset;

    // Start is called before the first frame update
    void Start()
    {
        m_animation = GetComponentInChildren<Animation>();
        call = GetComponentInChildren<AudioSource>();
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

        if(call != null)
        {
            call_timer -= Time.deltaTime;

            if (call_timer <= 0)
            {
                if (UnityEngine.Random.Range(0f, 1f) <= 0.2f)
                {
                    call.Play();
                }

                call_timer = 10f;
            }
        }
    }

    private void FixedUpdate()
    {
        float turn_speed = m_speed * UnityEngine.Random.Range(0.3f, 1f);
        Quaternion look_at = Quaternion.LookRotation(m_waypoint - transform.position);
        Quaternion correction = Quaternion.Euler(rotation_offset);

        transform.rotation = Quaternion.Slerp(transform.rotation, look_at * correction, Time.deltaTime * turn_speed);
        transform.position = Vector3.MoveTowards(transform.position, m_waypoint, m_speed * Time.deltaTime);
    }

    private Vector3 GetWaypoint()
    {
        return random_movement ? ai_manager.GetRandomValidPosition() : ai_manager.RandomWaypoint();
    }

    private void FindNewTarget()
    {
        m_last_waypoint = m_waypoint;
        m_waypoint = GetWaypoint();
        m_speed = UnityEngine.Random.Range(m_mix_max_speed.Item1, m_mix_max_speed.Item2);
    }

    private void ResolveCollisions()
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, (Quaternion.Inverse(Quaternion.Euler(rotation_offset)) * transform.forward).normalized, out hit, 10.0f))
        {
            if (hit.collider.tag == "Waypoint" || hit.collider.tag == "Terrain" || UnityEngine.Random.Range(1, 100) < 40)
            {
                FindNewTarget();
            }
        }
    }

    private void OnDrawGizmos()
    {
        Debug.DrawRay(transform.position, (Quaternion.Inverse(Quaternion.Euler(rotation_offset)) * transform.forward).normalized * 10, Color.green);
    }
}
