using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;

public class Server
{
    // Buffer from which read the packages. Ideally, this should be the
    // maximum size of a TCP packet, which is 65535. Ideally, we need to
    // receive less information than that!
    public static int dataBufferSize = 4096;
    public string ip = "127.0.0.1";
    public int port = 60260;

    public NetworkStream stream = null;
    private byte[] receiveBuffer;

    public bool useTCP = true;
    public TcpClient socket;

    public int desiredServerTickRate = 1;
    private int currentServerTickRate;
    private float desiredServerTimeStep;
    private float timeBetweenPackets = 0f;

    private bool awaitingData = false;

    // Start is called before the first frame update

    public Server(string ipadd, int p, int tickRate)
    {
        ip = ipadd;
        port = p;
        desiredServerTimeStep = (float)1 / tickRate;
    }

    public bool IsTcpGood()
    {
        return (socket != null) && (stream != null);
    }

    public async Task Connect()
    {
        socket = new TcpClient
        {
            ReceiveBufferSize = dataBufferSize,
            SendBufferSize = dataBufferSize
        };

        receiveBuffer = new byte[dataBufferSize];

        try
        {
            await socket.ConnectAsync(ip, port);
            stream = socket.GetStream();
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }
    }

    public async Task<string> AwaitAnyData()
    {
        int _byteLength = await stream.ReadAsync(receiveBuffer, 0, dataBufferSize);

        awaitingData = false;

        try
        {
            if (_byteLength <= 0)
            {
                if (socket.Connected)
                {
                    socket.Close();
                    // TODO: disconnect
                }
                return "";
            }

            byte[] _data = new byte[_byteLength];
            Array.Copy(receiveBuffer, _data, _byteLength);
            return System.Text.Encoding.Default.GetString(_data);
        }
        catch
        {
            if (socket.Connected)
            {
                socket.Close();
                // TODO: disconnect
            }
            return "";
        }
    }

    public void Update(float dt)
    {
        if(stream != null)
        {
            timeBetweenPackets += dt;

            if (awaitingData)
            {
                if (timeBetweenPackets >= desiredServerTimeStep && currentServerTickRate > 1)
                {
                    desiredServerTickRate--;
                }

                else if (timeBetweenPackets < desiredServerTimeStep && currentServerTickRate < desiredServerTickRate)
                {
                    desiredServerTickRate++;
                }

                return;
            }
        }
    }

    public bool GoodToSend()
    {
        return !awaitingData && (timeBetweenPackets >= desiredServerTimeStep);
    }

    public async Task SendDataAsync(byte[] _packet)
    {
        awaitingData = true;
        currentServerTickRate = (int)Mathf.Round(1f / timeBetweenPackets);
        timeBetweenPackets = 0f;

        try
        {
            if ((socket != null) && (stream != null))
            {
                await stream.WriteAsync(_packet, 0, _packet.Length);          
            }
        }
        catch (Exception _ex)
        {
            awaitingData = false;
            Debug.Log($"Error sending data to server via TCP: {_ex}");
            return;
        }
    }
}
