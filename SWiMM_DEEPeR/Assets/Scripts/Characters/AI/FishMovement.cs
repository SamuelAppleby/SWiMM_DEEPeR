using System;
using UnityEngine;

public class FishMovement : MonoBehaviour
{
    public float rover_check_timer = 5f;
    private float current_timer = 0f;
    public float distance_threshold = 100f;
    public bool random_movement = false;
    public FishSpawner ai_manager;
    private Vector3 m_waypoint;
    private Vector3 m_last_waypoint = new Vector3(0f, 0f, 0f);
    private Animation m_animation;
    private AudioSource call;
    private float call_timer = 10f;
    private float m_speed = 0f;
    public Tuple<float, float> m_mix_max_speed = new Tuple<float, float>(1, 7);
    private Collider m_collider;

    // For axis fixing import from fbx, 3dsmax etc
    public Vector3 rotation_offset = Vector3.zero;

    private Vector3 correct_forward;
    private Vector3 correct_up;
    private Vector3 correct_right;

    public Vector3 valid_movements;

    // Start is called before the first frame update
    void Start()
    {
        m_collider = GetComponentInChildren<Collider>();
        m_animation = GetComponentInChildren<Animation>();
        call = GetComponentInChildren<AudioSource>();
        FindNewTarget();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Minus))
        {
            transform.position = new Vector3(0, 100, 0);
        }

        if (Input.GetKeyDown(KeyCode.Equals))
        {
            transform.position = SimulationManager._instance.rover.transform.position;
        }

        if ((transform.position - SimulationManager._instance.rover.transform.position).magnitude > distance_threshold)
        {
            current_timer += Time.deltaTime;

            if(current_timer >= rover_check_timer)
            {
                FindNewTarget();
                current_timer = 0f;
            }
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
        DetectFutureCollisions();
        //Quaternion inverse = Quaternion.Inverse(Quaternion.Euler(rotation_offset));

        correct_forward = (/*inverse * */transform.forward).normalized;
        ////correct_right = (inverse * transform.right).normalized;
        ////correct_up = (inverse * transform.up).normalized;

        //Vector3 dir = m_waypoint - transform.position;

        //Vector3 angles = new Vector3(
        //    Vector3.SignedAngle(dir.normalized, correct_forward, -transform.right),
        //    Vector3.SignedAngle(dir.normalized, correct_forward, -transform.up),
        //    0) - rotation_offset;

        //angles *= Time.deltaTime * turn_speed;

        //transform.Rotate(angles);

        //find the vector pointing from our position to the target
        Vector3 _direction = (m_waypoint - transform.position).normalized;

        //create the rotation we need to be in to look at the target
        Quaternion _lookRotation = Quaternion.LookRotation(_direction);

        //rotate us over time according to speed until we are in the required rotation
        transform.rotation = Quaternion.Slerp(transform.rotation, _lookRotation, m_speed * 0.1f);

        //Quaternion q = transform.rotation;
        //q.eulerAngles = new Vector3(q.eulerAngles.x, q.eulerAngles.y, 0);
        //transform.rotation = q;

        transform.position += correct_forward * m_speed;

        if ((m_waypoint - transform.position).magnitude < 5)
        {
            FindNewTarget();
        }

        //Quaternion look_at = Quaternion.LookRotation(m_waypoint - transform.position);
        //Quaternion correction = Quaternion.Euler(rotation_offset);

        //transform.rotation = Quaternion.Slerp(transform.rotation, look_at * correction, Time.fixedDeltaTime * turn_speed);
        //transform.position = Vector3.MoveTowards(transform.position, m_waypoint, m_speed * Time.fixedDeltaTime);

        //if (Math.Abs(angle_z) > 5.0f)
        //{
        //    //transform.RotateAround(transform.position, correct_forwards, angle_z * Time.fixedDeltaTime * turn_speed);
        //}
        //if (transform.rotation.eulerAngles.z > 1.0f)
        //{
        //    transform.RotateAround(transform.position, -correct_forwards, 50 * Time.fixedDeltaTime * turn_speed);
        //}
        //if (transform.rotation.eulerAngles.z < -1.0f)
        //{
        //    transform.RotateAround(transform.position, correct_forwards, 50 * Time.deltaTime * turn_speed);
        //}

        //Quaternion lookOnLook = Quaternion.LookRotation(m_waypoint - transform.position);
        //transform.rotation = Quaternion.Slerp(transform.rotation, lookOnLook, Time.fixedDeltaTime * turn_speed);       // another option

        //transform.position = Vector3.MoveTowards(transform.position, m_waypoint, m_speed * Time.fixedDeltaTime);
    }

    private void FindNewTarget()
    {
        m_last_waypoint = m_waypoint;
        m_waypoint = random_movement ? ai_manager.GetRandomValidPosition(valid_movements) : ai_manager.RandomWaypoint();
        m_speed = UnityEngine.Random.Range(m_mix_max_speed.Item1, m_mix_max_speed.Item2);
    }

    private void DetectFutureCollisions()
    {
        RaycastHit[] hit = Physics.RaycastAll(transform.position, correct_forward, 10.0f);

        foreach(RaycastHit h in hit)
        {
            if (h.transform.gameObject.tag == "WaterSurface" || h.transform.gameObject.tag == "Player")
            {
                FindNewTarget();
            }
        }
    }

    void OnDrawGizmos()
    {
        Gizmos.color = new Color(0f, 1f, 0f, 0.3f);
        //Debug.DrawRay(transform.position, correct_forward * 10, Color.green);
        Debug.DrawRay(transform.position, correct_forward.normalized * 10, Color.green);
        Gizmos.DrawSphere(m_waypoint, 10);
    }
}
