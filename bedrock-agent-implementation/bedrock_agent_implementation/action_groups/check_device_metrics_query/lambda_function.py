import boto3
import json
import time
import os

def lambda_handler(event, context):
    try:
        # Extract necessary information from the input event
        print(event)
        action_group = event['actionGroup']
        api_path = event['apiPath']
        query_parameters = event['parameters'][0]
        sql_query = query_parameters['value']
        
        print(f"Executing SQL query: {sql_query}")

        # Specify your Athena database and output location
        database = os.getenv('ATHENA_DATABASE')
        output_location = os.getenv('ATHENA_OUTPUT_LOCATION')
        
        # Create Athena client
        athena_client = boto3.client('athena')

        # Run the Athena query
        response = athena_client.start_query_execution(
            QueryString=sql_query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': output_location}
        )

        # Get the query execution ID
        query_execution_id = response['QueryExecutionId']
        print(f"Query execution ID: {query_execution_id}")

        # Poll for the query execution status
        max_execution_time = 30  # Maximum execution time in seconds
        start_time = time.time()
        while (time.time() - start_time) < max_execution_time:
            query_status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)['QueryExecution']['Status']['State']
            print(f"Query status: {query_status}")
            if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(2)  # Wait for 2 seconds before checking again

        if query_status != 'SUCCEEDED':
            raise Exception(f"Query failed with status: {query_status}")

        # Get the query results
        results = athena_client.get_query_results(QueryExecutionId=query_execution_id)

        # Extract and return the results
        columns = [col_info['Name'] for col_info in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
        data = [dict(zip(columns, [item.get('VarCharValue', '') for item in row['Data']])) for row in results['ResultSet']['Rows'][1:]]
        
        print(f"Query results: {json.dumps(data)}")

        response_body = {
            'application/json': {
                'body': json.dumps(data)
            }
        }
        
        # Bedrock action group response format
        action_response = {
            "messageVersion": "1.0",
            "response": {
                'actionGroup': action_group,
                'apiPath': api_path,
                'httpMethod': event['httpMethod'],
                'httpStatusCode': 200,
                'responseBody': response_body
            }
        }
    
        return action_response

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "messageVersion": "1.0",
            "response": {
                'actionGroup': action_group,
                'apiPath': api_path,
                'httpMethod': event['httpMethod'],
                'httpStatusCode': 500,
                'responseBody': {
                    'application/json': {
                        'body': json.dumps({"error": str(e)})
                    }
                }
            }
        }
