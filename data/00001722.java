public class TestText implements Comparable<TestText> { String value; public TestText() { } public TestText( String string ) { this.value = string; } @Override public int compareTo( TestText o ) { if( o == null ) return 1; if( value == null && o.value == null ) return 0; if( value == null ) return -1; if( o.value == null ) return 1; return value.compareTo( o.value ); } @Override public String toString() { return value; } @Override public int hashCode() { if( value == null ) return 0; return value.hashCode(); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( object == null || getClass() != object.getClass() ) return false; TestText testText = (TestText) object; if( value != null ? !value.equals( testText.value ) : testText.value != null ) return false; return true; } }