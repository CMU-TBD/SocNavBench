#ifndef SOCKET_H
#define SOCKET_H

#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

// TODO: read these from the user_params.ini
#define SEND_ID "/tmp/socnavbench_joystick_recv" // default for SocNavBench
#define RECV_ID "/tmp/socnavbench_joystick_send"

using namespace std;

// Global socket information to use throughout the program
/* socket address of the sending-socket, to send commands to the robot */
struct sockaddr_un sender_addr = {AF_UNIX, SEND_ID};
/* file descriptor of the sending-socket, to send commands to the robot */
int sender_fd = 0;

/* socket address of the receiver-socket, to listen to the robot */
struct sockaddr_un receiver_addr = {AF_UNIX, RECV_ID};
/* file descriptor of the receiver-socket, to listen to the robot */
int receiver_fd = 0;

/**
 * @brief initializes the 'sender' (client) connection to the simulator
 * @param[in] robot_addr The address of the robot-sender socket
 * @param[in] robot_sender_fd The file descriptor of the robot-sender socket
 * @returns 0 if successful, -1 otherwise
 */
int init_send_conn(struct sockaddr_un &addr, int &sender_fd);

/**
 * @brief initializes the 'receiver' (server) connection to the simulator
 * @param[in] robot_addr The address of the robot-receiver socket
 * @param[in] robot_receiver_fd The file descriptor of the robot-receiver socket
 * @returns client_fd if successful (nonnegative), -1 otherwise
 */
int init_recv_conn(struct sockaddr_un &addr, int &receiver_fd);

/// TODO: read these in from the .ini param file instead of hardcoding

/** @brief Whether or not to include verbose printing */
const bool verbose = false;

/** @brief The number of times when the sockets were connected */
size_t num_connections = 0;

/**
 * @brief receives all the data from a client descriptor into a buffer at a certain rate
 * @param[in] conn_fd The file descriptor of the connection to receive from
 * @param[out] buffer The resulting buffer to write the data into
 * @param[in] buffer_amnt The maximum amount to recv() at a time
 * @returns response_len The number of bytes received from the socket connection
 */
int conn_recv(const int conn_fd, vector<char> &data, const int buf_amnt = 128)
{
    data.clear();
    int response_len = 0;
    char buffer[buf_amnt];
    while (true)
    {
        int chunk_amnt = recv(conn_fd, buffer, sizeof(buffer), 0);
        if (chunk_amnt <= 0)
            break;
        response_len += chunk_amnt;
        // append newly received chunk to overall data
        for (size_t i = 0; i < chunk_amnt; i++)
            data.push_back(buffer[i]);
    }
    return response_len;
}

/**
 * @brief Closes the input sockets manually
 * @param[in] send_fd The file descriptor of the "sending" socket
 * @param[in] recv_fd The file descriptor of the "receiving" socket
 */
void close_sockets(const int &send_fd, const int &recv_fd)
{
    close(send_fd);
    close(recv_fd);
}

/**
 * @brief sends a message (string) to the robot
 * @param[in] message The string to send to the robot, can be json string or literal
 * @returns 0 if successful, -1 otherwise
 */
int send_to_robot(const string &message)
{
    // create the TCP/IP socket and connect to the server (robot)
    if (init_send_conn(sender_addr, sender_fd) < 0)
        return -1;
    const void *buf = message.c_str();
    const size_t buf_len = message.size();
    int amnt_sent;
    if ((amnt_sent = send(sender_fd, buf, buf_len, 0)) < 0)
    {
        perror("\nsend() error: ");
        return -1;
    }
    close(sender_fd);
    if (verbose)
        cout << "sent " << amnt_sent << " bytes: "
             << "\"" << message << "\"" << endl;
    return 0;
}

/**
 * @brief waits for a response from the server and receives it all
 * @param[in] data The buffer to write the data into
 * @returns 0 if successful, -1 otherwise
 */
int listen_once(vector<char> &data)
{
    int client_fd;
    int addr_len = sizeof(receiver_addr);
    if ((client_fd = accept(receiver_fd, (struct sockaddr *)&receiver_addr, (socklen_t *)&addr_len)) < 0)
    {
        cout << "\033[31m"
             << "Unable to accept connection\n"
             << "\033[00m" << endl;
        return -1;
    }
    int response_len = conn_recv(client_fd, data);
    close(client_fd);
    if (verbose)
        cout << "\033[36m"
             << "Received " << response_len << " bytes from server"
             << "\033[00m" << endl;
    return 0;
}

/**
 * @brief initializes the 'sender' (client) connection to the simulator
 * @param[in] robot_addr The address of the robot-sender socket
 * @param[in] robot_sender_fd The file descriptor of the robot-sender socket
 * @returns 0 if successful, -1 otherwise
 */
int init_send_conn(struct sockaddr_un &robot_addr, int &robot_sender_fd)
{
    // "client" connection
    if ((robot_sender_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
    {
        perror("\nsocket() error: ");
        return -1;
    }
    if (connect(robot_sender_fd, (struct sockaddr *)&robot_addr, sizeof(robot_addr)) < 0)
    {
        cout << "\033[31m"
             << "Unable to connect to robot\n"
             << "\033[00m"
             << "Make sure you have a simulation instance running" << endl;
        return -1;
    }
    // success!
    if (verbose || num_connections < 1) // at least print the first time
        cout << "\033[32m"
             << "Robot <-- Joystick (sender) connection established"
             << "\033[00m" << endl;
    return 0;
}

/**
 * @brief initializes the 'receiver' (server) connection to the simulator
 * @param[in] robot_addr The address of the robot-receiver socket
 * @param[in] robot_receiver_fd The file descriptor of the robot-receiver socket
 * @returns client_fd if successful (nonnegative), -1 otherwise
 */
int init_recv_conn(struct sockaddr_un &robot_addr, int &robot_receiver_fd)
{
    int client;
    int opt = 1;
    if ((robot_receiver_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
    {
        perror("socket() error");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(robot_receiver_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0)
    {
        perror("setsockopt() error");
        exit(EXIT_FAILURE);
    }
    unlink(RECV_ID); // delete the UNIX socket file if still in use
    if (bind(robot_receiver_fd, (struct sockaddr *)&robot_addr, sizeof(robot_addr)) < 0)
    {
        perror("bind() error");
        exit(EXIT_FAILURE);
    }
    if (listen(robot_receiver_fd, 1) < 0)
    {
        perror("listen() error");
        exit(EXIT_FAILURE);
    }
    int addr_len = sizeof(robot_receiver_fd);
    if ((client = accept(robot_receiver_fd, (struct sockaddr *)&robot_addr, (socklen_t *)&addr_len)) < 0)
    {
        perror("accept() error");
        exit(EXIT_FAILURE);
    }
    // success!
    if (verbose || num_connections < 1) // at least print the first time
        cout << "\033[32m"
             << "Robot --> Joystick (receiver) connection established"
             << "\033[00m" << endl;
    // update count of the number of times the sockets have been connected
    num_connections++;
    // client should always be nonnegative integer
    return client;
}

#endif
