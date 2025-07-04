import asyncio
import aiomysql

async def test_xampp_connection():
    try:
        connection = await aiomysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',  # Empty for default XAMPP
            db='yourdbname'
        )
        print("✅ XAMPP MySQL connection successful!")
        connection.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test_xampp_connection())