using UnityEngine;
using System.Collections;

public class DolphinUserController : MonoBehaviour
{
    DolphinCharacter dolphinCharacter;

    void Start()
    {
        dolphinCharacter = GetComponent<DolphinCharacter>();
    }

    private void FixedUpdate()
    {
        if (Input.GetKeyDown(KeyCode.H))
        {
            dolphinCharacter.Hit();
        }

        if (Input.GetButtonDown("Fire1"))
        {
            dolphinCharacter.Bite();
        }

        if (Input.GetKeyDown(KeyCode.K))
        {
            dolphinCharacter.Death();
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            dolphinCharacter.Rebirth();
        }

        if (Input.GetKey(KeyCode.U))
        {
            dolphinCharacter.upDownSpeed = Mathf.Lerp(dolphinCharacter.upDownSpeed, 1f, Time.deltaTime);
        }

        if (Input.GetKey(KeyCode.N))
        {
            dolphinCharacter.upDownSpeed = Mathf.Lerp(dolphinCharacter.upDownSpeed, -1f, Time.deltaTime);
        }

        if (Input.GetKey(KeyCode.R))
        {
            dolphinCharacter.rollSpeed = Mathf.Lerp(dolphinCharacter.rollSpeed, 1f, Time.deltaTime);
        }

        if (Input.GetKey(KeyCode.T))
        {
            dolphinCharacter.rollSpeed = Mathf.Lerp(dolphinCharacter.rollSpeed, -1f, Time.deltaTime);

        }
        if (Input.GetKeyDown(KeyCode.B))
        {
            dolphinCharacter.Bite();
        }

        dolphinCharacter.upDownSpeed = Mathf.Lerp(dolphinCharacter.upDownSpeed, 0f, Time.deltaTime);
        dolphinCharacter.rollSpeed = Mathf.Lerp(dolphinCharacter.rollSpeed, 0f, Time.deltaTime);

        dolphinCharacter.forwardSpeed = Input.GetAxis("Horizontal");
        dolphinCharacter.turnSpeed = Input.GetAxis("Vertical");
    }
}
