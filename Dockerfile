FROM mcr.microsoft.com/devcontainers/python:3.11

# Install the components required by Visual Studio Code given the script found at https://github.com/microsoft/vscode-dev-containers/tree/main/script-library.
# This is disabled by default. Set INSTALL_VSCODE = 1 to enable.
ARG INSTALL_VSCODE=0
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install the (passwordless) SSH server
# Additionally, allow the user to call python directly
RUN if [ "${INSTALL_VSCODE}" = "0" ] ; \
    then apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/* \
    mkdir /var/run/sshd && mkdir -p /run/sshd \
    echo 'root:root' | chpasswd && \
    useradd -m ${USERNAME} && passwd -d ${USERNAME} && \
    usermod -aG www-data ${USERNAME} && \
    sed -i'' -e's/^#PermitRootLogin prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PasswordAuthentication yes$/PasswordAuthentication yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PermitEmptyPasswords no$/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config && \
    echo 'export PATH="/opt/conda/bin:$PATH"' >> /home/${USERNAME}/.bashrc ; \
    else usermod -aG www-data vscode ; \
    fi

# Install all the requirements in the requirements.txt
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install ipykernel && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp

# You may run all the code you require for customizing the environment here ...
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/keyrings/pgdg.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/pgdg.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        | sudo tee /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends postgresql-client-17 tmux \
    && rm -rf /var/lib/apt/lists/*


# Expose the SSH server
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]