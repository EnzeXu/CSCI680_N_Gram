public class PwDbOutputException extends Exception { public PwDbOutputException(String string) { super(string); } public PwDbOutputException(String string, Exception e) { super(string, e); } public PwDbOutputException(Exception e) { super(e); } private static final long serialVersionUID = 3321212743159473368L; }