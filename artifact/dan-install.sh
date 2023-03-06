# NOTE: install.sh for public use.
# For Dan developement

sudo apt-get update
sudo apt-get install tmux
cat <<EOF >> .tmux.conf
set -g history-limit 100000
set -sg escape-time 0
EOF
tmux

sudo apt install -y bash \
                  build-essential \
                  git \
                  curl \
                  wget \
                  libssl-dev \
                  ca-certificates \
                  gnupg \
                  apt-transport-https \
                  python3 \
                  python3-pip \
                  htop \
                  tree \
                  bash-completion \
                  vim \
                  tmux \
                  less \
                  sudo

python3 -m pip install --no-cache-dir --upgrade pip



sudo apt install zsh
wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
sh install.sh

cat <<EOF >> .zsh
export ZSH="/home/dans/.oh-my-zsh"
ZSH_THEME="gianu"
plugins=(git)
source $ZSH/oh-my-zsh.sh
export PATH="$HOME/.local/bin:$PATH"
export UPDATE_ZSH_DAYS=100 # update once every 60 days
export DISABLE_UPDATE_PROMPT=true # always reply Yes and automatically upgrade.
export DISABLE_AUTO_TITLE=true # Keep the window's name fixed in tmux
setopt no_share_history
unsetopt share_history
export LESS=-FRX
bindkey '\e[OH' beginning-of-line
bindkey '\e[OF' end-of-line
bindkey  "^[[1~"   beginning-of-line
bindkey  "^[[4~"   end-of-line
bindkey '^H' backward-kill-word
EOF


echo "export MY_TOKEN=" >> ~/.bashrc
git clone https://danielsnider:$MY_TOKEN@github.com/danielsnider/stable-diffusion-pytorch.git
git clone git@github.com:UofT-EcoSystem/algorithmic-efficiency.git
git clone https://danielsnider:$MY_TOKEN@github.com/danielsnider/hotline.git
cd hotline
pip3 install -e .
cd artifact
echo "source /home/dans/hotline/artifact/setup.sh" >> ~/.bashrc
bash build.sh


pip3 install --upgrade nvitop
echo 'export PATH="/home/dans/.local/bin:${PATH}"' >> .zshrc



curl -fsSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | sudo apt-key add -
curl -sL https://deb.nodesource.com/setup_18.x | sudo bash -
sudo apt install nodejs
npm -v
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install -y yarn
cd hotline/ui
yarn install
npm run dev

# FIX for NODEJS 18.x
curl -s https://deb.nodesource.com/gpgkey/nodesource.gpg.key | gpg --dearmor | tee /usr/share/keyrings/nodesource.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x focal main' > /etc/apt/sources.list.d/nodesource.list
echo 'deb-src [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x focal main' >> /etc/apt/sources.list.d/nodesource.list
chmod a+r /usr/share/keyrings/nodesource.gpg
apt update
apt install -y nodejs

git config --global user.name "Daniel Snider"
git config --global user.email "danielsnider12@gmail.com"
git config --global credential.helper store
git config --global core.editor "vim"
