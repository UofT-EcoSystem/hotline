FROM hotline_base_image:latest

WORKDIR /home/ubuntu/hotline/ui

# Create this folder so that it us owned by ubuntu even after docker mounts the results volume to this location. Without the folder will be owned by root.
RUN mkdir dist

# Instal nodejs
USER root
RUN curl -sL https://deb.nodesource.com/setup_18.x | bash -
RUN apt install nodejs
RUN npm -v
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN apt update
RUN apt install -y yarn

# Install UI dependencies
USER ubuntu
RUN yarn install

USER root
RUN git config --global --add safe.directory /home/ubuntu/hotline
CMD git pull  && npm run dev