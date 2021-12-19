/*
   
Data and Database Management with SQL
   
*/

--------------------------------------------------------------------------------
/*				                 Banking DDL           		  		          */
-------------------------------------------------------------------------------
CREATE TABLE branch 
( 
branch_name VARCHAR (35) NOT NULL , 
branch_city VARCHAR (15), 
assets NUMERIC(12,2)CHECK (assets > 0.00), 
CONSTRAINT branch_pkey PRIMARY KEY (branch_name) 
);


CREATE TABLE customer 
( 
ID  INTEGER , 
customer_name  VARCHAR(25), 
customer_street VARCHAR(30), 
customer_city VARCHAR(20), 
CONSTRAINT customer_pkey PRIMARY KEY (ID) );


CREATE TABLE loan 
(
loan_number  INTEGER , 
branch_name VARCHAR (35) , 
amount NUMERIC(12,2) CHECK (amount >0.00),
CONSTRAINT loan_pkey PRIMARY KEY ( loan_number), 
CONSTRAINT loan_fkey FOREIGN KEY(branch_name) REFERENCES branch ON DELETE CASCADE );


CREATE TABLE  borrower 
(
ID INTEGER , 
loan_number INTEGER, 
CONSTRAINT borrower_pkey PRIMARY KEY (ID, loan_number), 
CONSTRAINT borrower_fkey FOREIGN KEY (loan_number) REFERENCES loan ( loan_number) ON UPDATE CASCADE
);


CREATE TABLE account 
(
account_number INTEGER , 
branch_name VARCHAR(35) ,
balance NUMERIC(10,2) DEFAULT 0.00 , 
CONSTRAINT account_pkey PRIMARY KEY ( account_number), 
CONSTRAINT account_fkey FOREIGN KEY(branch_name) REFERENCES branch(branch_name)
);


CREATE TABLE depositor 
(
ID INTEGER , 
account_number INTEGER, 
CONSTRAINT depositor_pkey PRIMARY KEY(ID , account_number)
);

--------------------------------------------------------------------------------
/*				                  Question 1           		  		          */
--------------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION Alem_04_monthlyPayment(p_amount NUMERIC(18, 2), apr NUMERIC(18, 6), years INTEGER)
RETURNS NUMERIC(18, 2)
LANGUAGE plpgsql
AS
$$
	DECLARE 
		a_amount NUMERIC(18, 2);
		n NUMERIC(18, 6);
		i_amount NUMERIC(18, 6);
	BEGIN
		i_amount := apr/12;
		n := years*12; 
		a_amount := (p_amount * (i_amount + (i_amount / (POWER(1+i_amount, n) - 1))));
		RETURN a_amount;
	END;
$$;

--------------------------------------------------------------------------------
/*				                  Question 2           		  		          */
--------------------------------------------------------------------------------
    ------------------------------- Part (a) -----------------------------

SELECT ID, customer_name 
FROM customer As cust
WHERE EXISTS (
	SELECT * FROM borrower AS bw
	WHERE bw.ID = cust.ID)
AND NOT EXISTS(	
SELECT * FROM depositor AS dep
WHERE dep.ID = cust.ID)

 ------------------------------- Part (b) ------------------------------

SELECT DISTINCT ID, customer_name
FROM customer AS cu
WHERE cu.customer_street = (SELECT  cus.customer_street
	  FROM customer as cus
	  WHERE cus.ID = 12345)
	  AND cu.customer_city = (SELECT  cus.customer_city
	  FROM customer as cus
	  WHERE cus.ID = 12345)
-- Customer ID 12345 is also included ---

  ------------------------------- Part (c) ------------------------------
		
  SELECT DISTINCT branch_name
  From account NATURAL JOIN depositor NATURAL JOIN customer
  WHERE customer_city = 'Harrison'

  ------------------------------- Part (d) ------------------------------

	
	SELECT DISTINCT cus.customer_name
	FROM customer AS cus
WHERE EXISTS 
     (SELECT *
      FROM branch AS bra, account AS acc, depositor AS dep
      WHERE dep.ID = cus.ID AND
			acc.account_number = dep.account_number AND
	  		bra.branch_name = acc.branch_name AND
	  		bra.branch_city = 'Brooklyn')
    															
--------------------------------------------------------------------------------
/*				                  Question 3           		  		          */
--------------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION Alem_04_bankTriggerFunction() 
RETURNS TRIGGER 
LANGUAGE plpgsql
AS
$$        
    
        BEGIN
              DELETE FROM depositor WHERE depositor.ID NOT IN (SELECT ID FROM depositor 
								WHERE account_number <> OLD.account_number);
              RETURN NEW;
               
        END;
		$$;
    


CREATE TRIGGER Alem_04_bankTrigger 
AFTER DELETE ON account
FOR EACH ROW 
EXECUTE PROCEDURE Alem_04_bankTriggerFunction();

--------------------------------------------------------------------------------
/*				                  Question 4           		  		          */
--------------------------------------------------------------------------------

CREATE TEMPORARY TABLE instructor_course_nums (
                ID VARCHAR(5),
                name VARCHAR(20),
                tot_courses INTEGER
                );

CREATE OR REPLACE PROCEDURE Alem_04__insCourseNumsProc(INOUT i_ID VARCHAR(5))
         LANGUAGE plpgsql
         AS
         $$
            DECLARE
                c_count INTEGER := 0;
                ins_Name VARCHAR(20) := '';
            BEGIN
                 SELECT COUNT(t.course_id) INTO c_count
                 FROM teaches AS t INNER JOIN instructor AS i ON t.ID = i.ID
                 WHERE t.ID = Alem_04__insCourseNumsProc.i_ID;
                 SELECT i.name INTO ins_Name
                 FROM instructor AS i
                 WHERE i.ID = Alem_04__insCourseNumsProc.i_ID;
                                        
                 IF EXISTS (SELECT ID 
                            FROM instructor_course_nums
                            WHERE ID = Alem_04__insCourseNumsProc.i_ID
                            ) THEN       
                            UPDATE instructor_course_nums
                            SET tot_courses = c_count
                            WHERE ID = Alem_04__insCourseNumsProc.i_ID;
                 ELSE   
                     INSERT INTO instructor_course_nums (ID, name, tot_courses)
                     VALUES (Alem_04__insCourseNumsProc.i_ID, ins_Name, c_count);
                 END IF;
             END;
         $$
