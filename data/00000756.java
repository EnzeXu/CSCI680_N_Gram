public class SearchParameters implements Cloneable { public static final SearchParameters DEFAULT = new SearchParameters ( ) ; public String searchString ; public boolean regularExpression = false ; public boolean searchInTitles = true ; public boolean searchInUserNames = true ; public boolean searchInPasswords = false ; public boolean searchInUrls = true ; public boolean searchInGroupNames = false ; public boolean searchInNotes = true ; public boolean ignoreCase = true ; public boolean ignoreExpired = false ; public boolean respectEntrySearchingDisabled = true ; public boolean excludeExpired = false ; @ Override public Object clone ( ) { try { return super . clone ( ) ; } catch ( CloneNotSupportedException e ) { return null ; } } public void setupNone ( ) { searchInTitles = false ; searchInUserNames = false ; searchInPasswords = false ; searchInUrls = false ; searchInGroupNames = false ; searchInNotes = false ; } }