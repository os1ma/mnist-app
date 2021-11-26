CREATE TABLE `models` (
  `id` int PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `tag` varchar(255) NOT NULL,
  `created_at` timestamp DEFAULT current_timestamp
);

CREATE TABLE `images` (
  `id` int PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `original_image_path` varchar(255) NOT NULL,
  `preprocessed_image_path` varchar(255) NOT NULL,
  `created_at` timestamp DEFAULT current_timestamp
);

CREATE TABLE `predictions` (
  `id` int PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `model_id` int NOT NULL,
  `image_id` int NOT NULL,
  `result` json NOT NULL,
  `created_at` timestamp DEFAULT current_timestamp
);

ALTER TABLE `predictions` ADD FOREIGN KEY (`model_id`) REFERENCES `models` (`id`);

ALTER TABLE `predictions` ADD FOREIGN KEY (`image_id`) REFERENCES `images` (`id`);
