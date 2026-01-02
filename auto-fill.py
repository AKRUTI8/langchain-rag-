from playwright.sync_api import sync_playwright
import time
import os

# Configuration - Update these values
CONFIG = {
    'resume_path': 'D:\internship\chatbot\Akruti-patel.pdf',  # Path to your resume PDF
    'cover_letter_path': './cover-letter.pdf',  # Path to your cover letter PDF
    'job_url': 'https://www.coupang.jobs/en/jobs/?gh_jid=7301665',
    
    # Resume data - Extract this from your resume or fill manually
    'resume_data': {
        'first_name': 'John',
        'last_name': 'Doe',
        'email': 'john.doe@example.com',
        'phone': '+1234567890',
        'location': 'Seoul, South Korea',
        'city': 'Seoul',
        'country': 'South Korea',
        'linkedin': 'https://linkedin.com/in/johndoe',
        'website': 'https://johndoe.com',
        'current_company': 'Tech Corp',
        'current_title': 'Software Engineer',
        'years_of_experience': '5',
        'education': 'Bachelor of Science in Computer Science',
        'university': 'Seoul National University',
        'graduation_year': '2018'
    }
}


def fill_text_field(page, field_names, value):
    """Helper function to fill text fields using multiple strategies"""
    for name in field_names:
        try:
            # Try by name attribute
            field = page.locator(f'input[name*="{name}" i]').first
            if field.is_visible(timeout=500):
                field.fill(value)
                return True
        except:
            pass
        
        try:
            # Try by id attribute
            field = page.locator(f'input[id*="{name}" i]').first
            if field.is_visible(timeout=500):
                field.fill(value)
                return True
        except:
            pass
        
        try:
            # Try by placeholder
            field = page.locator(f'input[placeholder*="{name}" i]').first
            if field.is_visible(timeout=500):
                field.fill(value)
                return True
        except:
            pass
    
    return False


def upload_file(page, field_names, file_path):
    """Helper function to upload files"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    for name in field_names:
        try:
            # Try by name attribute
            file_input = page.locator(f'input[type="file"][name*="{name}" i]').first
            if file_input.is_visible(timeout=500):
                file_input.set_input_files(file_path)
                print(f"Uploaded: {file_path}")
                return True
        except:
            pass
        
        try:
            # Try by id attribute
            file_input = page.locator(f'input[type="file"][id*="{name}" i]').first
            if file_input.is_visible(timeout=500):
                file_input.set_input_files(file_path)
                print(f"Uploaded: {file_path}")
                return True
        except:
            pass
    
    # Try any visible file input
    try:
        file_inputs = page.locator('input[type="file"]').all()
        for input_field in file_inputs:
            if input_field.is_visible():
                input_field.set_input_files(file_path)
                print(f"Uploaded: {file_path}")
                return True
    except:
        print(f"Could not upload file: {file_path}")
    
    return False


def fill_job_application():
    """Main function to automate job application"""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # Set to True for headless mode
            slow_mo=100  # Slow down actions for visibility
        )
        
        context = browser.new_context()
        page = context.new_page()
        
        try:
            print('Navigating to job page...')
            page.goto(CONFIG['job_url'], wait_until='networkidle')
            time.sleep(2)
            
            # Accept cookies
            print('Accepting cookies...')
            cookie_selectors = [
                'button:has-text("Accept all")',
                'button:has-text("Accept All")',
                'button:has-text("Accept")',
                '[id*="accept"]',
                '[class*="accept"]'
            ]
            
            for selector in cookie_selectors:
                try:
                    cookie_btn = page.locator(selector).first
                    if cookie_btn.is_visible(timeout=2000):
                        cookie_btn.click()
                        print('Cookies accepted')
                        time.sleep(1)
                        break
                except:
                    continue
            
            # Click Apply Now button
            print('Clicking Apply Now...')
            apply_selectors = [
                'button:has-text("Apply Now")',
                'a:has-text("Apply Now")',
                '[class*="apply"]',
                '#apply-button'
            ]
            
            for selector in apply_selectors:
                try:
                    apply_btn = page.locator(selector).first
                    if apply_btn.is_visible(timeout=2000):
                        apply_btn.click()
                        print('Apply Now clicked')
                        time.sleep(2)
                        break
                except:
                    continue
            
            # Wait for application form to load
            time.sleep(3)
            
            # Fill text inputs
            print('Filling form fields...')
            resume_data = CONFIG['resume_data']
            
            fill_text_field(page, ['first_name', 'firstName', 'first-name'], resume_data['first_name'])
            fill_text_field(page, ['last_name', 'lastName', 'last-name'], resume_data['last_name'])
            fill_text_field(page, ['email', 'email_address', 'e-mail'], resume_data['email'])
            fill_text_field(page, ['phone', 'telephone', 'mobile'], resume_data['phone'])
            fill_text_field(page, ['location', 'address', 'city'], resume_data['location'])
            fill_text_field(page, ['linkedin', 'linked_in'], resume_data['linkedin'])
            fill_text_field(page, ['website', 'portfolio', 'url'], resume_data['website'])
            fill_text_field(page, ['company', 'current_company'], resume_data['current_company'])
            fill_text_field(page, ['title', 'job_title', 'position'], resume_data['current_title'])
            fill_text_field(page, ['experience', 'years'], resume_data['years_of_experience'])
            fill_text_field(page, ['education', 'degree'], resume_data['education'])
            fill_text_field(page, ['university', 'school', 'college'], resume_data['university'])
            
            # Handle textareas (cover letter, additional info)
            textareas = page.locator('textarea').all()
            for textarea in textareas:
                if textarea.is_visible():
                    textarea.fill('I am excited about this opportunity and believe my skills align well with the requirements.')
            
            # Upload resume
            print('Uploading resume...')
            upload_file(page, ['resume', 'cv'], CONFIG['resume_path'])
            
            # Upload cover letter
            print('Uploading cover letter...')
            upload_file(page, ['cover', 'letter', 'cover_letter'], CONFIG['cover_letter_path'])
            
            # Handle all dropdowns/select elements
            print('Handling dropdown selections...')
            selects = page.locator('select').all()
            for select in selects:
                if select.is_visible():
                    options = select.locator('option').all()
                    if len(options) > 0:
                        # Try to find "Yes" option first
                        selected_option = False
                        for option in options:
                            text = option.text_content()
                            if text and ('yes' in text.lower() or text.lower() == 'y'):
                                select.select_option(label=text)
                                selected_option = True
                                break
                        
                        # If no "Yes" option, select the second option (skip first which is usually placeholder)
                        if not selected_option and len(options) > 1:
                            select.select_option(index=1)
            
            # Handle radio buttons (select "Yes" if available)
            print('Handling radio buttons...')
            radio_groups = page.locator('input[type="radio"]').all()
            processed_groups = set()
            
            for radio in radio_groups:
                name = radio.get_attribute('name')
                if name and name not in processed_groups:
                    processed_groups.add(name)
                    
                    # Try to find and select "Yes" option
                    try:
                        yes_radio = page.locator(f'input[type="radio"][name="{name}"][value*="yes" i], input[type="radio"][name="{name}"][value*="y" i]').first
                        if yes_radio.is_visible(timeout=500):
                            yes_radio.check()
                            continue
                    except:
                        pass
                    
                    # Select first radio button in the group
                    try:
                        radio.check()
                    except:
                        pass
            
            # Handle checkboxes that might be required
            print('Handling checkboxes...')
            checkboxes = page.locator('input[type="checkbox"]').all()
            for checkbox in checkboxes:
                label = checkbox.get_attribute('aria-label') or ''
                checkbox_id = checkbox.get_attribute('id') or ''
                
                # Check boxes that seem to be agreements/confirmations
                if ('agree' in label.lower() or 
                    'confirm' in label.lower() or 
                    'terms' in label.lower() or 
                    'agree' in checkbox_id.lower()):
                    if not checkbox.is_checked():
                        checkbox.check()
            
            print('Form filling completed!')
            print('Ready to submit. Waiting 3 seconds...')
            time.sleep(3)
            
            # Submit the form
            print('Submitting application...')
            submit_selectors = [
                'button[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Submit Application")',
                'input[type="submit"]',
                '[class*="submit"]'
            ]
            
            for selector in submit_selectors:
                try:
                    submit_btn = page.locator(selector).first
                    if submit_btn.is_visible(timeout=2000):
                        submit_btn.click()
                        print('Application submitted!')
                        break
                except:
                    continue
            
            # Wait to see the result
            time.sleep(5)
            print('Process completed!')
            
        except Exception as error:
            print(f'Error during automation: {error}')
            # Take screenshot on error
            page.screenshot(path='error-screenshot.png', full_page=True)
        
        finally:
            browser.close()


if __name__ == '__main__':
    fill_job_application()