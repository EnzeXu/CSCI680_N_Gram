public class TestConstants { @Deprecated public static final RegexSplitter TAB_SPLITTER = new RegexSplitter( "\t" ); public static final int[] APACHE_COMMON_GROUPS = new int[]{1, 2, 3, 4, 5, 6}; public static final Fields APACHE_COMMON_GROUP_FIELDS = new Fields( "ip", "time", "method", "event", "status", "size" ); public static final String APACHE_COMMON_REGEX = "^([^ ]*) +[^ ]* +[^ ]* +\\[([^]]*)\\] +\\\"([^ ]*) ([^ ]*) [^ ]*\\\" ([^ ]*) ([^ ]*).*$"; public static final RegexParser APACHE_COMMON_PARSER = new RegexParser( APACHE_COMMON_GROUP_FIELDS, APACHE_COMMON_REGEX, APACHE_COMMON_GROUPS ); public static final String APACHE_DATE_FORMAT = "dd/MMM/yyyy:HH:mm:ss Z"; @Deprecated public static final DateParser APACHE_DATE_PARSER = new DateParser( APACHE_DATE_FORMAT ); @Deprecated public static final FieldJoiner TAB_JOINER = new FieldJoiner( "\t" ); }