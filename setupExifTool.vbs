Option Explicit

Dim objXMLHTTP, objStream, strDownloadURL, strZipPath, strExtractTo, objFSO, strExeSource, strExeTarget, objShell

' Create the Shell object and FileSystemObject
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objXMLHTTP = CreateObject("MSXML2.XMLHTTP")
Set objStream = CreateObject("ADODB.Stream")

' URL of the zip file
strDownloadURL = "https://exiftool.org/exiftool-12.85.zip"

' Local path to save the downloaded zip
strZipPath = objShell.ExpandEnvironmentStrings("%TEMP%") & "\exiftool-12.85.zip"

' Folder to extract contents to (ensure the directory has appropriate permissions)
strExtractTo = objShell.ExpandEnvironmentStrings("%APPDATA%") & "\ExifTool"

' Download the file
objXMLHTTP.Open "GET", strDownloadURL, False
objXMLHTTP.Send

If objXMLHTTP.Status = 200 Then
    ' Save the binary data stream
    With objStream
        .Type = 1 'adTypeBinary
        .Open
        .Write objXMLHTTP.ResponseBody
        .SaveToFile strZipPath, 2 'adSaveCreateOverWrite
        .Close
    End With

    ' Use Shell.Application to unzip
    Dim objApp, objFolder
    Set objApp = CreateObject("Shell.Application")
    Set objFolder = objApp.NameSpace(strZipPath)

    ' Create target directory if it doesn't exist
    If Not objFSO.FolderExists(strExtractTo) Then
        objFSO.CreateFolder(strExtractTo)
    End If

    ' Extract files
    objApp.NameSpace(strExtractTo).CopyHere objFolder.Items, 4 + 16  ' 4: No progress dialog, 16: Respond with "Yes to All" for any dialogs

    ' Wait for extraction to finish
    Do While objApp.NameSpace(strExtractTo).Items.Count < objFolder.Items.Count
        WScript.Sleep 100
    Loop

    ' Rename the extracted executable
    strExeSource = strExtractTo & "\exiftool(-k).exe"
    strExeTarget = strExtractTo & "\exiftool.exe"

    If objFSO.FileExists(strExeSource) Then
        objFSO.MoveFile strExeSource, strExeTarget
    End If

    ' Set environment variable
    Dim strVarName, strVarValue
    strVarName = "exiftool"
    strVarValue = strExeTarget

    objShell.Environment("USER")(strVarName) = strVarValue
    WScript.Echo "Environment variable 'exiftool' set to " & strVarValue
Else
    WScript.Echo "Failed to download file. Status: " & objXMLHTTP.Status
End If

' Clean up
Set objXMLHTTP = Nothing
Set objStream = Nothing
Set objShell = Nothing
Set objFSO = Nothing