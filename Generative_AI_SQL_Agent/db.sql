CREATE DATABASE IF NOT EXISTS store_db;
USE store_db;

CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(100),
    price DECIMAL(10, 2)
);

CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    order_date DATE,
    quantity INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Insert Sample Data
INSERT INTO customers (name, email) VALUES
('Alice Smith', 'alice@example.com'),
('Bob Johnson', 'bob@example.com'),
('Charlie Davis', 'charlie@example.com');

INSERT INTO products (product_name, price) VALUES
('Laptop', 1200.00),
('Wireless Mouse', 25.50),
('Mechanical Keyboard', 85.00);

INSERT INTO orders (customer_id, product_id, order_date, quantity) VALUES
(1, 1, '2023-10-01', 1), -- Alice bought 1 Laptop
(1, 2, '2023-10-05', 2), -- Alice bought 2 Mice
(2, 3, '2023-10-10', 1), -- Bob bought 1 Keyboard
(3, 1, '2023-10-12', 2); -- Charlie bought 2 Laptops