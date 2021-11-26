create database app;

create table app.posts (
  id int primary key auto_increment,
  message text not null,
  created_at timestamp not null default current_timestamp
);

create user 'app'@'%' identified by 'password';

grant select,insert,update,delete on app.* to 'app'@'%';
