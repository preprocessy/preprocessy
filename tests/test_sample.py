# format for test filename
# test_filename.py

# function names should start with 'test_'
def test_hello_world():
    s = "Hello World"
    assert s == "Hello World"


def test_feature_name():
    s = "Hello World"
    assert s == 10


# class names for tests should start with 'Test'
class TestSomeFeature:
    def test_one(self):
        assert True == False

    def test_two(self):
        assert True != True
